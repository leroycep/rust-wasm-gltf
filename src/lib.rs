use gltf::Mesh;
use std::cell::RefCell;
use std::collections::HashSet;
use std::mem::size_of;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{WebGl2RenderingContext, WebGlProgram, WebGlShader, WebGlUniformLocation};

pub fn compile_shader(
    context: &WebGl2RenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, String> {
    let shader = context
        .create_shader(shader_type)
        .ok_or_else(|| String::from("Unable to create shader object"))?;
    context.shader_source(&shader, source);
    context.compile_shader(&shader);

    if context
        .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(context
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| String::from("Unknown error creating shader")))
    }
}

pub fn link_program(
    context: &WebGl2RenderingContext,
    vert_shader: &WebGlShader,
    frag_shader: &WebGlShader,
) -> Result<WebGlProgram, String> {
    let program = context
        .create_program()
        .ok_or_else(|| String::from("Unable to create shader object"))?;

    context.attach_shader(&program, vert_shader);
    context.attach_shader(&program, frag_shader);
    context.link_program(&program);

    if context
        .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        Err(context
            .get_program_info_log(&program)
            .unwrap_or_else(|| String::from("Unknown error creating program object")))
    }
}

#[wasm_bindgen]
pub fn load_gltf_model(gltf_bytes: Box<[u8]>) -> Result<(), JsValue> {
    log(&format!(
        "load_gltf_model loading {} bytes",
        gltf_bytes.len()
    ));

    let (document, buffers, _images) =
        gltf::import_slice(&gltf_bytes).map_err(|err| format!("error parsing gltf: {}", err))?;

    let mut meshes = Vec::new();

    for mesh in document.meshes() {
        log(&format!("reading mesh; name = {:?}", mesh.name()));

        meshes.push(load_mesh(&mesh, &buffers));

        log(&format!(
            "Loaded {} vertices, {} indices; min = {}, max = {}",
            meshes.last().unwrap().vertices.len() / 3,
            meshes.last().unwrap().indices.len(),
            meshes.last().unwrap().min,
            meshes.last().unwrap().max,
        ));
    }

    let mut root_nodes = Vec::new();
    if let Some(scene) = document.default_scene() {
        for node in scene.nodes() {
            root_nodes.push(load_scene_nodes(&node));
        }
    }

    let mut animations = Vec::new();
    for animation in document.animations() {
        animations.push(load_animation(&animation, &buffers));
    }

    Ok(display_model(&meshes, root_nodes, animations)?)
}

static QUIT_RENDERING: AtomicBool = AtomicBool::new(false);

fn display_model(
    meshes: &[VertexMesh],
    model: Vec<VertexMeshInstance>,
    animations: Vec<Animation>,
) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let performance = window.performance().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into::<web_sys::HtmlCanvasElement>()?;

    let context = canvas
        .get_context("webgl2")?
        .unwrap()
        .dyn_into::<WebGl2RenderingContext>()?;

    let vert_shader = compile_shader(
        &context,
        WebGl2RenderingContext::VERTEX_SHADER,
        r#"
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;
        attribute vec3 position;
        attribute vec3 aNormal;

        varying vec3 vert_pos;
        varying vec3 vert_normal;

        void main() {
            vert_pos = (model * vec4(position, 1.0)).xyz;
            vert_normal = aNormal;
            gl_Position = projection * view * model * vec4(position, 1.0);
        }
    "#,
    )?;
    let frag_shader = compile_shader(
        &context,
        WebGl2RenderingContext::FRAGMENT_SHADER,
        r#"
        #ifdef GL_ES
        precision mediump float;
        #endif
        uniform vec3 light_pos;
        uniform vec3 view_pos;

        varying vec3 vert_pos;
        varying vec3 vert_normal;

        const float ambient_strength = 0.4;
        const float specular_strength = 0.5;
        const vec3 light_color = vec3(1.0);
        const vec3 object_color = vec3(136.0 / 255.0);

        void main() {
            vec3 normal = normalize(vert_normal);
            vec3 ambient = ambient_strength * light_color;
        
            vec3 light_dir = normalize(light_pos - vert_pos);
            float diffuse_strength = max(dot(normal, light_dir), 0.0);
            vec3 diffuse = diffuse_strength * light_color;

            vec3 view_dir = normalize(view_pos - vert_pos);
            vec3 reflect_dir = reflect(-light_dir, normal);
            float specular_intensity = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
            vec3 specular = specular_strength * specular_intensity * light_color;
        
            gl_FragColor = vec4((ambient + diffuse + specular) * object_color, 1.0);
        }
    "#,
    )?;
    let program = link_program(&context, &vert_shader, &frag_shader)?;
    context.use_program(Some(&program));
    let projection_uniform = context
        .get_uniform_location(&program, "projection")
        .ok_or("Could not get projection uniform location")?;
    let transform_uniform = context
        .get_uniform_location(&program, "view")
        .ok_or("Could not get transform uniform location")?;
    let model_transform_uniform = context
        .get_uniform_location(&program, "model")
        .ok_or("Could not get transform uniform location")?;
    let light_pos_uniform = context
        .get_uniform_location(&program, "light_pos")
        .ok_or("Could not get light_pos uniform location")?;
    let view_pos_uniform = context
        .get_uniform_location(&program, "view_pos")
        .ok_or("Could not get view_pos uniform location")?;

    let line_vert_shader = compile_shader(
        &context,
        WebGl2RenderingContext::VERTEX_SHADER,
        r#"
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;

        attribute vec3 position;
        attribute vec3 aNormal;

        void main() {
            vec3 model_pos = (model * vec4(position, 1.0)).xyz;
            vec3 offset_vec = normalize((model * vec4(aNormal, 0.0)).xyz);
            vec3 offset_pos = model_pos + offset_vec * 0.0001;
            gl_Position = projection * view * vec4(offset_pos, 1.0);
        }
    "#,
    )?;
    let line_frag_shader = compile_shader(
        &context,
        WebGl2RenderingContext::FRAGMENT_SHADER,
        r#"
        #ifdef GL_ES
        precision mediump float;
        #endif

        void main() {
            gl_FragColor = vec4(0.0, 0.0, 0.8, 0.0);
        }
    "#,
    )?;
    let line_program = link_program(&context, &line_vert_shader, &line_frag_shader)?;
    let line_projection_uniform = context
        .get_uniform_location(&line_program, "projection")
        .ok_or_else(|| JsValue::from_str("Could not get projection uniform location"))?;
    let line_view_uniform = context
        .get_uniform_location(&line_program, "view")
        .ok_or_else(|| JsValue::from_str("Could not get view uniform location"))?;
    let line_model_uniform = context
        .get_uniform_location(&line_program, "model")
        .ok_or_else(|| JsValue::from_str("Could not get model uniform location"))?;

    let mut gpu_meshes = Vec::new();
    let mut line_gpu_meshes = Vec::new();
    // Upload meshes
    for mesh in meshes {
        gpu_meshes.push(upload_mesh_to_gpu(&context, mesh)?);
        line_gpu_meshes.push(upload_mesh_to_gpu(
            &context,
            &line_mesh_from_triangle_mesh(mesh),
        )?);
    }

    let animation_index = 1;
    log(&format!(
        "playing animations[{}] = {}",
        animation_index,
        animations[animation_index].name.as_ref().unwrap()
    ));
    log(&format!(
        "animations[{}] = {:?}",
        animation_index,
        &animations[animation_index]
            .node_values
            .keys()
            .collect::<Vec<_>>()
    ));
    for root_node in &model {
        root_node.dump_debug_info();
    }

    let mut min = glam::vec3(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = glam::vec3(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for mesh_instance in &model {
        log(&format!("transform = {}", mesh_instance.local_transform()));
        let (mesh_min, mesh_max) = animations[animation_index].min_max_bounds(
            &meshes,
            mesh_instance,
            glam::Mat4::IDENTITY,
            0.0,
        );
        min = min.min(mesh_min);
        max = max.max(mesh_max);
    }
    let size = (max - min).max_element();
    let center = (min + max) / 2.0;
    log(&format!("size = {}, center = {}", size, center));

    let model_scale = glam::f32::Mat4::from_scale(glam::Vec3::splat(1.0 / size));
    let model_offset = glam::f32::Mat4::from_translation(-center);

    let projection =
        glam::f32::Mat4::perspective_infinite_rh(f32::to_radians(45.0), 640.0 / 480.0, 0.01);

    let light_pos = glam::vec3(1000.0, 1000.0, 1000.0);

    let f = Rc::new(RefCell::new(None));
    let g = f.clone();

    QUIT_RENDERING.store(false, Ordering::SeqCst);
    let mut i: u64 = 0;
    let start_time = (performance.now() / 1000.0) as f32;
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        if QUIT_RENDERING.load(Ordering::SeqCst) {
            log(&format!("Stopping rendering"));

            let _ = f.borrow_mut().take();
            return;
        }

        let current_time = (performance.now() / 1000.0) as f32;
        let time = current_time - start_time;

        context.enable(WebGl2RenderingContext::CULL_FACE);
        context.cull_face(WebGl2RenderingContext::BACK);

        context.enable(WebGl2RenderingContext::DEPTH_TEST);

        context.clear_color(0.0, 0.0, 0.0, 1.0);
        context.clear(
            WebGl2RenderingContext::COLOR_BUFFER_BIT | WebGl2RenderingContext::DEPTH_BUFFER_BIT,
        );

        let theta = time * (2.0 * std::f32::consts::PI) / 5.0;
        let radius = 1.5;
        let camera_pos = glam::vec3(theta.sin() * radius, 0.5, theta.cos() * radius);

        let view_matrix = glam::f32::Mat4::look_at_rh(
            camera_pos,
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(0.0, 1.0, 0.0),
        );

        // Render faces
        context.use_program(Some(&program));
        context.uniform_matrix4fv_with_f32_array(
            Some(&projection_uniform),
            false,
            &projection.to_cols_array(),
        );
        context.uniform_matrix4fv_with_f32_array(
            Some(&transform_uniform),
            false,
            &view_matrix.to_cols_array(),
        );
        context.uniform3fv_with_f32_array(Some(&light_pos_uniform), &light_pos.to_array());
        context.uniform3fv_with_f32_array(Some(&view_pos_uniform), &camera_pos.to_array());

        for root_node in &model {
            animations[animation_index].render(
                &context,
                &program,
                &model_transform_uniform,
                root_node,
                &gpu_meshes,
                model_scale * model_offset,
                time,
            );
        }

        // Render lines
        context.use_program(Some(&line_program));
        context.uniform_matrix4fv_with_f32_array(
            Some(&line_projection_uniform),
            false,
            &projection.to_cols_array(),
        );
        context.uniform_matrix4fv_with_f32_array(
            Some(&line_view_uniform),
            false,
            &view_matrix.to_cols_array(),
        );
        for root_node in &model {
            animations[animation_index].render_line(
                &context,
                &line_program,
                &line_model_uniform,
                root_node,
                &line_gpu_meshes,
                model_scale * model_offset,
                time,
            );
        }

        let i = &mut i;
        *i += 1;

        request_animation_frame(f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}

#[wasm_bindgen]
pub fn quit_rendering() {
    log(&format!("Setting QUIT_RENDERING to true"));

    QUIT_RENDERING.store(true, Ordering::SeqCst);
}

fn window() -> web_sys::Window {
    web_sys::window().expect("no global `window` exists")
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    window()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("should register `requestAnimationFrame` OK");
}

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

}

struct Animation {
    name: Option<String>,
    node_values: std::collections::BTreeMap<usize, NodeAnimationValues>,
}

impl Animation {
    pub fn min_max_bounds(
        &self,
        meshes: &[VertexMesh],
        mesh_instance: &VertexMeshInstance,
        parent_transform: glam::Mat4,
        time_since_start: f32,
    ) -> (glam::Vec3, glam::Vec3) {
        let mut min = glam::vec3(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = glam::vec3(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        let animation_transform = self.transform_at_time_for_node(time_since_start, mesh_instance);

        let transform = parent_transform * animation_transform;

        if let Some(mesh_index) = mesh_instance.mesh_index {
            min = min.min(transform.transform_point3(meshes[mesh_index].min));
            max = max.max(transform.transform_point3(meshes[mesh_index].max));
        }
        for child in &mesh_instance.children {
            let (mesh_min, mesh_max) =
                self.min_max_bounds(meshes, child, transform, time_since_start);
            min = min.min(mesh_min);
            max = max.max(mesh_max);
        }

        (min, max)
    }

    pub fn render(
        &self,
        context: &WebGl2RenderingContext,
        program: &WebGlProgram,
        model_transform_uniform: &WebGlUniformLocation,
        mesh_instance: &VertexMeshInstance,
        meshes: &[GpuMesh],
        parent_transform: glam::f32::Mat4,
        time_since_start: f32,
    ) {
        let animation_transform = self.transform_at_time_for_node(time_since_start, mesh_instance);

        let model_transform = parent_transform * animation_transform;

        if let Some(mesh_index) = mesh_instance.mesh_index {
            let gpu_mesh = &meshes[mesh_index];
            context.uniform_matrix4fv_with_f32_array(
                Some(&model_transform_uniform),
                false,
                &model_transform.to_cols_array(),
            );

            context.bind_buffer(
                WebGl2RenderingContext::ARRAY_BUFFER,
                Some(&gpu_mesh.vertex_buffer),
            );
            context.bind_buffer(
                WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
                Some(&gpu_mesh.index_buffer),
            );

            let position_attribute = context.get_attrib_location(program, "position");
            let normal_attribute = context.get_attrib_location(program, "aNormal");

            context.vertex_attrib_pointer_with_i32(
                position_attribute as u32,
                3,
                WebGl2RenderingContext::FLOAT,
                false,
                (6 * size_of::<f32>()) as i32,
                0,
            );
            context.enable_vertex_attrib_array(position_attribute as u32);
            context.vertex_attrib_pointer_with_i32(
                normal_attribute as u32,
                3,
                WebGl2RenderingContext::FLOAT,
                false,
                (6 * size_of::<f32>()) as i32,
                (3 * size_of::<f32>()) as i32,
            );
            context.enable_vertex_attrib_array(normal_attribute as u32);

            context.draw_elements_with_i32(
                WebGl2RenderingContext::TRIANGLES,
                gpu_mesh.num_indices,
                WebGl2RenderingContext::UNSIGNED_SHORT,
                0,
            );
        }

        for child in mesh_instance.children.iter() {
            self.render(
                context,
                program,
                model_transform_uniform,
                child,
                meshes,
                model_transform,
                time_since_start,
            );
        }
    }

    pub fn render_line(
        &self,
        context: &WebGl2RenderingContext,
        program: &WebGlProgram,
        model_transform_uniform: &WebGlUniformLocation,
        mesh_instance: &VertexMeshInstance,
        meshes: &[GpuMesh],
        parent_transform: glam::f32::Mat4,
        time_since_start: f32,
    ) {
        let animation_transform = self.transform_at_time_for_node(time_since_start, mesh_instance);

        let model_transform = parent_transform * animation_transform;

        if let Some(mesh_index) = mesh_instance.mesh_index {
            let gpu_mesh = &meshes[mesh_index];
            context.uniform_matrix4fv_with_f32_array(
                Some(&model_transform_uniform),
                false,
                &model_transform.to_cols_array(),
            );

            context.bind_buffer(
                WebGl2RenderingContext::ARRAY_BUFFER,
                Some(&gpu_mesh.vertex_buffer),
            );
            context.bind_buffer(
                WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
                Some(&gpu_mesh.index_buffer),
            );

            let position_attribute = context.get_attrib_location(program, "position");
            let normal_attribute = context.get_attrib_location(program, "aNormal");

            context.vertex_attrib_pointer_with_i32(
                position_attribute as u32,
                3,
                WebGl2RenderingContext::FLOAT,
                false,
                (6 * size_of::<f32>()) as i32,
                0,
            );
            context.enable_vertex_attrib_array(position_attribute as u32);
            context.vertex_attrib_pointer_with_i32(
                normal_attribute as u32,
                3,
                WebGl2RenderingContext::FLOAT,
                false,
                (6 * size_of::<f32>()) as i32,
                (3 * size_of::<f32>()) as i32,
            );
            context.enable_vertex_attrib_array(normal_attribute as u32);

            context.draw_elements_with_i32(
                WebGl2RenderingContext::LINES,
                gpu_mesh.num_indices,
                WebGl2RenderingContext::UNSIGNED_SHORT,
                0,
            );
        } else {
            // For now, render mesh[0] just to check that the skeleton is being animated
            let gpu_mesh = &meshes[0];
            context.uniform_matrix4fv_with_f32_array(
                Some(&model_transform_uniform),
                false,
                &model_transform.to_cols_array(),
            );

            context.bind_buffer(
                WebGl2RenderingContext::ARRAY_BUFFER,
                Some(&gpu_mesh.vertex_buffer),
            );
            context.bind_buffer(
                WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
                Some(&gpu_mesh.index_buffer),
            );

            let position_attribute = context.get_attrib_location(program, "position");
            let normal_attribute = context.get_attrib_location(program, "aNormal");

            context.vertex_attrib_pointer_with_i32(
                position_attribute as u32,
                3,
                WebGl2RenderingContext::FLOAT,
                false,
                (6 * size_of::<f32>()) as i32,
                0,
            );
            context.enable_vertex_attrib_array(position_attribute as u32);
            context.vertex_attrib_pointer_with_i32(
                normal_attribute as u32,
                3,
                WebGl2RenderingContext::FLOAT,
                false,
                (6 * size_of::<f32>()) as i32,
                (3 * size_of::<f32>()) as i32,
            );
            context.enable_vertex_attrib_array(normal_attribute as u32);

            context.draw_elements_with_i32(
                WebGl2RenderingContext::LINES,
                gpu_mesh.num_indices,
                WebGl2RenderingContext::UNSIGNED_SHORT,
                0,
            );
        }

        for child in mesh_instance.children.iter() {
            self.render_line(
                context,
                program,
                model_transform_uniform,
                child,
                meshes,
                model_transform,
                time_since_start,
            );
        }
    }

    pub fn transform_at_time_for_node(
        &self,
        time_since_start: f32,
        mesh_instance: &VertexMeshInstance,
    ) -> glam::Mat4 {
        let mut translation = mesh_instance.translation;
        let mut rotation = mesh_instance.rotation;
        let mut scale = mesh_instance.scale;

        if let Some(node) = self.node_values.get(&mesh_instance.node_index) {
            let current_time = time_since_start % node.translations_timeline.last().unwrap();

            for (time_index, next_time) in node.translations_timeline.iter().enumerate() {
                if *next_time > current_time && time_index > 0 {
                    let prev_time = node.translations_timeline[time_index - 1];
                    let fraction_towards_next_time =
                        (current_time - prev_time) / (next_time - prev_time);

                    let next_translation = node.translations[time_index];
                    let prev_translation = node.translations[time_index - 1];
                    translation = prev_translation
                        + fraction_towards_next_time * (next_translation - prev_translation);
                    break;
                }
            }

            for (time_index, next_time) in node.rotations_timeline.iter().enumerate() {
                if *next_time > current_time && time_index > 0 {
                    let prev_time = node.rotations_timeline[time_index - 1];
                    let fraction_towards_next_time =
                        (current_time - prev_time) / (next_time - prev_time);

                    let next_rotation = node.rotations[time_index];
                    let prev_rotation = node.rotations[time_index - 1];
                    rotation = prev_rotation
                        + (next_rotation - prev_rotation) * fraction_towards_next_time;
                    break;
                }
            }

            for (time_index, next_time) in node.scales_timeline.iter().enumerate() {
                if *next_time > current_time && time_index > 0 {
                    let prev_time = node.scales_timeline[time_index - 1];
                    let fraction_towards_next_time =
                        (current_time - prev_time) / (next_time - prev_time);

                    let next_scale = node.scales[time_index];
                    let prev_scale = node.scales[time_index - 1];
                    scale = prev_scale + fraction_towards_next_time * (next_scale - prev_scale);
                    break;
                }
            }
        }

        glam::Mat4::from_scale_rotation_translation(scale, rotation, translation)
    }
}

#[derive(Clone, Debug, Default)]
struct NodeAnimationValues {
    translations_timeline: Vec<f32>,
    translations: Vec<glam::Vec3>,

    rotations_timeline: Vec<f32>,
    rotations: Vec<glam::Quat>,

    scales_timeline: Vec<f32>,
    scales: Vec<glam::Vec3>,
}

fn load_animation(gltf_animation: &gltf::Animation, buffers: &[gltf::buffer::Data]) -> Animation {
    log(&format!("animation name = {:?}", gltf_animation.name()));
    let mut node_animation_values: std::collections::BTreeMap<usize, NodeAnimationValues> =
        std::collections::BTreeMap::new();
    for channel in gltf_animation.channels() {
        log(&format!("{:?}", channel.sampler().interpolation()));

        let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));

        let mut inputs = Vec::new();
        for input in reader.read_inputs().unwrap() {
            inputs.push(input);
        }

        match reader.read_outputs() {
            Some(gltf::animation::util::ReadOutputs::Translations(translations)) => {
                let mut outputs = Vec::new();
                for translation in translations {
                    outputs.push(glam::Vec3::from_array(translation));
                }

                let values = node_animation_values
                    .entry(channel.target().node().index())
                    .or_default();
                values.translations_timeline = inputs;
                values.translations = outputs;
            }
            Some(gltf::animation::util::ReadOutputs::Rotations(rotations)) => {
                let mut outputs = Vec::new();
                for rotation in rotations.into_f32() {
                    outputs.push(glam::Quat::from_array(rotation));
                }

                let values = node_animation_values
                    .entry(channel.target().node().index())
                    .or_default();
                values.rotations_timeline = inputs;
                values.rotations = outputs;
            }
            Some(gltf::animation::util::ReadOutputs::Scales(scales)) => {
                let mut outputs = Vec::new();
                for scale in scales {
                    outputs.push(glam::Vec3::from_array(scale));
                }

                let values = node_animation_values
                    .entry(channel.target().node().index())
                    .or_default();
                values.scales_timeline = inputs;
                values.scales = outputs;
            }
            Some(gltf::animation::util::ReadOutputs::MorphTargetWeights(morph_target_weights)) => {
                let mut outputs = Vec::new();
                for weight in morph_target_weights.into_f32() {
                    outputs.push(weight);
                }
                log(&format!("morph weights = {:?}", &outputs));
            }
            None => log("it's nothing"),
        }
    }

    Animation {
        name: gltf_animation.name().map(|s| s.into()),
        node_values: node_animation_values,
    }
}

#[derive(Debug)]
struct VertexMeshInstance {
    node_index: usize,
    mesh_index: Option<usize>,

    translation: glam::f32::Vec3,
    rotation: glam::f32::Quat,
    scale: glam::f32::Vec3,

    children: Vec<VertexMeshInstance>,
}

impl VertexMeshInstance {
    pub fn local_transform(&self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    pub fn min_max_bounds(
        &self,
        meshes: &[VertexMesh],
        parent_transform: glam::Mat4,
    ) -> (glam::Vec3, glam::Vec3) {
        let mut min = glam::vec3(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = glam::vec3(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        if let Some(mesh_index) = self.mesh_index {
            let transform = parent_transform * self.local_transform();
            min = min.min(transform.transform_point3(meshes[mesh_index].min));
            max = max.max(transform.transform_point3(meshes[mesh_index].max));
        }
        for child in &self.children {
            let (mesh_min, mesh_max) =
                child.min_max_bounds(meshes, parent_transform * self.local_transform());
            min = min.min(mesh_min);
            max = max.max(mesh_max);
        }

        (min, max)
    }

    pub fn dump_debug_info(&self) {
        log(&format!(
            "node_id = {}, mesh_index = {:?}, num_children = {}",
            self.node_index,
            self.mesh_index,
            self.children.len()
        ));

        for child in &self.children {
            child.dump_debug_info();
        }
    }
}

fn load_scene_nodes(gltf_node: &gltf::Node) -> VertexMeshInstance {
    log(&format!("Node name = {:?}", gltf_node.name()));
    if let Some(_cam) = gltf_node.camera() {
        log("Node has a camera! Ignored, as cameras are not yet implemented");
    }

    let (translation, rotation, scale) = gltf_node.transform().decomposed();

    let mut children = Vec::new();
    for child in gltf_node.children() {
        children.push(load_scene_nodes(&child));
    }

    VertexMeshInstance {
        node_index: gltf_node.index(),
        mesh_index: gltf_node.mesh().map(|mesh| mesh.index()),
        translation: glam::Vec3::from_array(translation),
        rotation: glam::Quat::from_array(rotation),
        scale: glam::Vec3::from_array(scale),
        children,
    }
}

struct VertexMesh {
    min: glam::f32::Vec3,
    max: glam::f32::Vec3,
    vertices: Vec<f32>,
    indices: Vec<u16>,
}

fn load_mesh(gltf_mesh: &Mesh, buffers: &[gltf::buffer::Data]) -> VertexMesh {
    let mut min = glam::f32::Vec3::ZERO;
    let mut max = glam::f32::Vec3::ZERO;
    let mut vertices = Vec::<f32>::new();
    let mut indices = Vec::<u16>::new();

    for prim in gltf_mesh.primitives() {
        log(&format!(
            "\tprimitive = {:?}; bounds = {:?}",
            prim.mode(),
            prim.bounding_box()
        ));

        let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
        for index in reader.read_indices().unwrap().into_u32() {
            indices.push(index as u16);
        }
        if let Some(normal_reader) = reader.read_normals() {
            log(&format!("Model has normals"));
            for (pos, normal) in reader.read_positions().unwrap().zip(normal_reader) {
                vertices.push(pos[0]);
                vertices.push(pos[1]);
                vertices.push(pos[2]);
                vertices.push(normal[0]);
                vertices.push(normal[1]);
                vertices.push(normal[2]);

                min = min.min(glam::Vec3::from_array(pos));
                max = max.max(glam::Vec3::from_array(pos));
            }
        } else {
            for pos in reader.read_positions().unwrap() {
                vertices.push(pos[0]);
                vertices.push(pos[1]);
                vertices.push(pos[2]);
                vertices.push(0.0);
                vertices.push(0.0);
                vertices.push(0.0);

                min = min.min(glam::Vec3::from_array(pos));
                max = max.max(glam::Vec3::from_array(pos));
            }

            // Automatically calculate normals
            for i in 0..indices.len() / 3 {
                let triangle_indices = &indices[i * 3 as usize..][0..3];
                let triangle_pos = [
                    glam::Vec3::from_slice(&vertices[(6 * triangle_indices[0]) as usize..][0..3]),
                    glam::Vec3::from_slice(&vertices[(6 * triangle_indices[1]) as usize..][0..3]),
                    glam::Vec3::from_slice(&vertices[(6 * triangle_indices[2]) as usize..][0..3]),
                ];
                let face_normal = (triangle_pos[0] - triangle_pos[1])
                    .cross(triangle_pos[1] - triangle_pos[2])
                    .to_array();
                for index in triangle_indices {
                    let offset = (6 * index) as usize;
                    let triangle_verts = &mut vertices[offset..][3..6];
                    for component in 0..3 {
                        triangle_verts[component] += face_normal[component];
                    }
                }
            }

            for i in 0..vertices.len() / 6 {
                let offset = (6 * i) as usize;
                let normal_sums = &mut vertices[offset..][3..6];

                let normal = glam::Vec3::from_slice(&normal_sums).normalize();
                normal.write_to_slice(normal_sums);
            }
        }
    }

    VertexMesh {
        min,
        max,
        vertices,
        indices,
    }
}

fn line_mesh_from_triangle_mesh(mesh: &VertexMesh) -> VertexMesh {
    let mut lines = HashSet::new();

    for i in 0..mesh.indices.len() / 3 {
        let a = mesh.indices[i * 3];
        let b = mesh.indices[i * 3 + 1];
        let c = mesh.indices[i * 3 + 2];

        let line_ab = (a.min(b), a.max(b));
        let line_bc = (b.min(c), b.max(c));
        let line_ca = (c.min(a), c.max(a));
        lines.insert(line_ab);
        lines.insert(line_bc);
        lines.insert(line_ca);
    }

    let mut indices = Vec::<u16>::new();
    for line in lines {
        indices.push(line.0);
        indices.push(line.1);
    }

    VertexMesh {
        min: mesh.min,
        max: mesh.max,
        vertices: mesh.vertices.clone(),
        indices,
    }
}

struct GpuMesh {
    vertex_buffer: web_sys::WebGlBuffer,
    index_buffer: web_sys::WebGlBuffer,
    num_indices: i32,
}

fn upload_mesh_to_gpu(
    context: &WebGl2RenderingContext,
    mesh: &VertexMesh,
) -> Result<GpuMesh, JsValue> {
    let vertex_buffer = context.create_buffer().ok_or("failed to create buffer")?;
    context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&vertex_buffer));

    // Note that `Float32Array::view` is somewhat dangerous (hence the
    // `unsafe`!). This is creating a raw view into our module's
    // `WebAssembly.Memory` buffer, but if we allocate more pages for ourself
    // (aka do a memory allocation in Rust) it'll cause the buffer to change,
    // causing the `Float32Array` to be invalid.
    //
    // As a result, after `Float32Array::view` we have to be very careful not to
    // do any memory allocations before it's dropped.
    unsafe {
        let vert_array = js_sys::Float32Array::view(&mesh.vertices);

        context.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &vert_array,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    let index_buffer = context
        .create_buffer()
        .ok_or("failed to create index buffer")?;
    context.bind_buffer(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        Some(&index_buffer),
    );

    unsafe {
        let index_array = js_sys::Uint16Array::view(&mesh.indices);

        context.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
            &index_array,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    Ok(GpuMesh {
        vertex_buffer,
        index_buffer,
        num_indices: mesh.indices.len() as i32,
    })
}
