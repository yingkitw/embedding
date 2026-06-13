use ndarray::{Array1, Array2};

/// Abstraction over compute backends for matrix operations.
///
/// The library ships with a CPU backend by default. GPU acceleration
/// can be enabled via the `gpu` feature flag (experimental).
pub trait Backend {
    /// Initializes an embedding matrix with the given shape.
    fn init_embeddings(&self, vocab_size: usize, dim: usize) -> Array2<f32>;

    /// Computes the dot product of two vectors.
    fn dot(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32;

    /// Adds vector `b` scaled by `scale` into vector `a` in-place.
    fn add_scaled(&self, a: &mut Array1<f32>, b: &Array1<f32>, scale: f32);

    /// Matrix multiplication: `c = a * b` where `a` is (m, k) and `b` is (k, n).
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;

    /// Returns the backend name for diagnostics.
    fn name(&self) -> &'static str;
}

/// CPU backend using ndarray (default).
#[derive(Default)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackend {
    fn init_embeddings(&self, vocab_size: usize, dim: usize) -> Array2<f32> {
        use ndarray::Array;
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (dim as f32).sqrt();
        Array::from_shape_fn((vocab_size, dim), |_| rng.gen_range(-0.5..0.5) * scale)
    }

    fn dot(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    fn add_scaled(&self, a: &mut Array1<f32>, b: &Array1<f32>, scale: f32) {
        for (ai, bi) in a.iter_mut().zip(b.iter()) {
            *ai += bi * scale;
        }
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        a.dot(b)
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}

/// Returns the default backend (CPU).
pub fn default_backend() -> Box<dyn Backend> {
    Box::new(CpuBackend::new())
}

/// Attempts to create the best available backend (GPU if compiled with
/// the `gpu` feature and a device is available, otherwise CPU).
pub fn best_backend() -> Box<dyn Backend> {
    #[cfg(feature = "gpu")]
    {
        if let Ok(gpu) = GpuBackend::new() {
            return Box::new(gpu);
        }
    }
    Box::new(CpuBackend::new())
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ndarray::Array;
    use wgpu::util::DeviceExt;

    /// GPU compute backend using wgpu compute shaders.
    ///
    /// Runs matrix operations on the GPU via Vulkan, Metal, or DX12.
    /// Falls back to CPU if no GPU is available.
    pub struct GpuBackend {
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    impl GpuBackend {
        /// Creates a new GPU backend, or returns an error if no GPU is available.
        pub fn new() -> Result<Self, String> {
            let instance = wgpu::Instance::default();
            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .ok_or("No GPU adapter found")?;

            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("embedding-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            ))
            .map_err(|e| format!("Failed to create GPU device: {}", e))?;

            Ok(Self {
                device,
                queue,
            })
        }

        fn create_buffer(&self, data: &[f32], usage: wgpu::BufferUsages) -> wgpu::Buffer {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu-buffer"),
                contents: bytemuck::cast_slice(data),
                usage,
            })
        }

        fn create_uniform_buffer(&self, data: &[u8]) -> wgpu::Buffer {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniform-buffer"),
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        }

        fn read_buffer(&self, buffer: &wgpu::Buffer, len: usize) -> Vec<f32> {
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: buffer.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read-encoder"),
            });
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
            self.queue.submit(Some(encoder.finish()));

            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.device.poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data)[..len].to_vec();
            drop(data);
            staging.unmap();
            result
        }

        fn dispatch_1d(&self, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, count: u32) {
            let workgroups = (count + 255) / 256;
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute-encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("compute-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }

        fn dispatch_2d(&self, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, wx: u32, wy: u32) {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute-encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("compute-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(wx, wy, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }
    }

    impl Backend for GpuBackend {
        fn init_embeddings(&self, vocab_size: usize, dim: usize) -> Array2<f32> {
            // GPU random init is complex; generate on CPU and upload is one-time cost
            CpuBackend::new().init_embeddings(vocab_size, dim)
        }

        fn dot(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
            let len = a.len() as u32;
            let buf_a = self.create_buffer(a.as_slice().unwrap(), wgpu::BufferUsages::STORAGE);
            let buf_b = self.create_buffer(b.as_slice().unwrap(), wgpu::BufferUsages::STORAGE);
            let buf_out = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dot-out"),
                size: ((len + 255) / 256 * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dot-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dot-bind"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
                ],
            });

            // Recreate pipeline with proper layout
            let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("dot-shader"),
                source: wgpu::ShaderSource::Wgsl(DOT_SHADER.into()),
            });
            let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dot-pipeline-layout"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("dot-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

            self.dispatch_1d(&pipeline, &bind_group, len);

            let partials = self.read_buffer(&buf_out, ((len + 255) / 256) as usize);
            partials.iter().sum()
        }

        fn add_scaled(&self, a: &mut Array1<f32>, b: &Array1<f32>, scale: f32) {
            let len = a.len() as u32;
            let buf_a = self.create_buffer(a.as_slice().unwrap(), wgpu::BufferUsages::STORAGE);
            let buf_b = self.create_buffer(b.as_slice().unwrap(), wgpu::BufferUsages::STORAGE);
            let buf_scale = self.create_uniform_buffer(bytemuck::cast_slice(&[scale]));

            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("add-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("add-bind"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_scale.as_entire_binding() },
                ],
            });

            let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("add-shader"),
                source: wgpu::ShaderSource::Wgsl(ADD_SCALED_SHADER.into()),
            });
            let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("add-pipeline-layout"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("add-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

            self.dispatch_1d(&pipeline, &bind_group, len);

            let result = self.read_buffer(&buf_a, a.len());
            for (i, v) in result.iter().enumerate() {
                a[i] = *v;
            }
        }

        fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
            let m = a.nrows() as u32;
            let k = a.ncols() as u32;
            let n = b.ncols() as u32;

            let buf_a = self.create_buffer(a.as_slice().unwrap(), wgpu::BufferUsages::STORAGE);
            let buf_b = self.create_buffer(b.as_slice().unwrap(), wgpu::BufferUsages::STORAGE);
            let buf_c = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("matmul-c"),
                size: (m * n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let buf_dims = self.create_uniform_buffer(bytemuck::cast_slice(&[m, k, n, 0u32]));

            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("matmul-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul-bind"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_dims.as_entire_binding() },
                ],
            });

            let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul-shader"),
                source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
            });
            let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("matmul-pipeline-layout"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matmul-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

            let wx = (n + 15) / 16;
            let wy = (m + 15) / 16;
            self.dispatch_2d(&pipeline, &bind_group, wx, wy);

            let result = self.read_buffer(&buf_c, (m * n) as usize);
            Array::from_shape_vec((m as usize, n as usize), result).unwrap()
        }

        fn name(&self) -> &'static str {
            "gpu"
        }
    }

    const ADD_SCALED_SHADER: &str = r#"
        @group(0) @binding(0) var<storage, read_write> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<uniform> scale: f32;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i = gid.x;
            if (i >= arrayLength(&a)) { return; }
            a[i] = a[i] + b[i] * scale;
        }
    "#;

    const DOT_SHADER: &str = r#"
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> partials: array<f32>;

        var<workgroup> local_sums: array<f32, 256>;

        @compute @workgroup_size(256)
        fn main(
            @builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>,
            @builtin(workgroup_id) wgid: vec3<u32>
        ) {
            let global_i = gid.x;
            let local_i = lid.x;

            var prod = 0.0;
            if (global_i < arrayLength(&a)) {
                prod = a[global_i] * b[global_i];
            }
            local_sums[local_i] = prod;
            workgroupBarrier();

            // Tree reduction
            for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
                if (local_i < stride) {
                    local_sums[local_i] = local_sums[local_i] + local_sums[local_i + stride];
                }
                workgroupBarrier();
            }

            if (local_i == 0u) {
                partials[wgid.x] = local_sums[0];
            }
        }
    "#;

    const MATMUL_SHADER: &str = r#"
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> c: array<f32>;
        @group(0) @binding(3) var<uniform> dims: vec4<u32>;

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let row = gid.y;
            let col = gid.x;
            let m = dims.x;
            let k = dims.y;
            let n = dims.z;

            if (row >= m || col >= n) { return; }

            var sum = 0.0;
            for (var i = 0u; i < k; i = i + 1u) {
                sum = sum + a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    "#;
}

#[cfg(feature = "gpu")]
pub use gpu::GpuBackend;

