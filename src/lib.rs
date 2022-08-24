use gltf::Mesh;
use std::cell::RefCell;
use std::mem::size_of;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{WebGl2RenderingContext, WebGlProgram, WebGlShader};

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
            "Loaded {} vertices, {} indices; bounds = {:#?}",
            meshes.last().unwrap().vertices.len() / 3,
            meshes.last().unwrap().indices.len(),
            meshes.last().unwrap().bounds,
        ));
    }

    let mut model = Vec::new();
    let mut camera = None;
    if let Some(scene) = document.default_scene() {
        for node in scene.nodes() {
            model.extend(load_scene_nodes(&node));
            camera = load_scene_camera(&node);
        }
    }

    Ok(display_model(&meshes, model, camera)?)
}

static QUIT_RENDERING: AtomicBool = AtomicBool::new(false);

fn display_model(
    meshes: &[VertexMesh],
    model: Vec<VertexMeshInstance>,
    camera_opt: Option<Camera>,
) -> Result<(), JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
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
        varying vec3 view_space_position;

        void main() {
            vert_pos = (model * vec4(position, 1.0)).xyz;
            vert_normal = aNormal;
            view_space_position = (view * model * vec4(position, 1.0)).xyz;
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
        varying vec3 vert_pos;
        varying vec3 vert_normal;
        varying vec3 view_space_position;

        const float ambient_strength = 0.1;
        const float specular_strength = 0.5;
        const vec3 light_color = vec3(0.8, 0.8, 0.8);
        const vec3 object_color = vec3(136.0 / 255.0);
        const float alpha = 1.0;
        const vec3 light_pos = vec3(-100.0, 100.0, 0.0);

        void main() {
            vec3 normal = normalize(vert_normal);
            vec3 ambient = ambient_strength * light_color;
        
            vec3 light_dir = normalize(light_pos - vert_pos);
            float diffuse_strength = max(dot(normal, light_dir), 0.0);
            vec3 diffuse = diffuse_strength * light_color;

            vec3 view_dir = normalize(view_space_position);
            vec3 reflect_vec = reflect(-light_dir, normal);
            float specular_intensity = pow(max(dot(reflect_vec, view_dir), 0.0), 32.0);
            vec3 specular = specular_strength * specular_intensity * light_color;
        
            gl_FragColor = vec4((ambient + diffuse + specular) * object_color, alpha);
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

    let line_frag_shader = compile_shader(
        &context,
        WebGl2RenderingContext::FRAGMENT_SHADER,
        r#"
        void main() {
            gl_FragColor = vec4(0.0, 0.0, 0.4, 0.0);
        }
    "#,
    )?;
    let line_program = link_program(&context, &vert_shader, &line_frag_shader)?;
    let line_transform_uniform = context
        .get_uniform_location(&line_program, "view")
        .ok_or_else(|| JsValue::from_str("Could not get transform uniform location"))?;

    let mut furthest: f32 = 0.01;

    let mut gpu_meshes = Vec::new();
    // Upload meshes
    for mesh in meshes {
        gpu_meshes.push(upload_mesh_to_gpu(&context, mesh)?);

        for pos in mesh.bounds {
            for coord in pos {
                furthest = furthest.max(coord);
            }
        }
    }

    let camera = camera_opt.unwrap_or(Camera {
        translation: glam::f32::vec3(0.0, 0.0, furthest),
        rotation: glam::f32::vec4(0.0, 0.0, 0.0, 1.0),
        aspect: 640.0 / 480.0,
        fov: f32::to_radians(45.0),
        near: 0.01,
        far: 100.0,
    });
    let projection =
        glam::f32::Mat4::perspective_infinite_rh(camera.fov, camera.aspect, camera.near);
    log(&format!("camera = {:?}", camera));

    let f = Rc::new(RefCell::new(None));
    let g = f.clone();

    QUIT_RENDERING.store(false, Ordering::SeqCst);
    let mut i: u64 = 0;
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        if QUIT_RENDERING.load(Ordering::SeqCst) {
            log(&format!("Stopping rendering"));

            let _ = f.borrow_mut().take();
            return;
        }

        context.enable(WebGl2RenderingContext::CULL_FACE);
        context.cull_face(WebGl2RenderingContext::BACK);

        context.enable(WebGl2RenderingContext::DEPTH_TEST);

        context.clear_color(0.0, 0.0, 0.0, 1.0);
        context.clear(
            WebGl2RenderingContext::COLOR_BUFFER_BIT | WebGl2RenderingContext::DEPTH_BUFFER_BIT,
        );

        let theta = (i as f32) / 60.0;
        let radius = camera.translation.length();
        let view_matrix = glam::f32::Mat4::look_at_rh(
            glam::vec3(theta.sin() * radius, 0.0, theta.cos() * radius),
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(0.0, 1.0, 0.0),
        );

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

        for instance in &model {
            let gpu_mesh = &gpu_meshes[instance.index];

            // Render faces
            context.uniform_matrix4fv_with_f32_array(
                Some(&model_transform_uniform),
                false,
                &instance.transform,
            );
            context.bind_buffer(
                WebGl2RenderingContext::ARRAY_BUFFER,
                Some(&gpu_mesh.vertex_buffer),
            );
            context.bind_buffer(
                WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
                Some(&gpu_mesh.index_buffer),
            );

            let position_attribute = context.get_attrib_location(&program, "position");
            let normal_attribute = context.get_attrib_location(&program, "aNormal");

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

        i += 1;

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

fn mat4_mul(left: [f32; 4 * 4], right: [f32; 4 * 4]) -> [f32; 4 * 4] {
    let mut res = [0.0; 4 * 4];
    for i in 0..4 {
        for j in 0..4 {
            let res_index = j * 4 + i;
            for k in 0..4 {
                let left_index = i * 4 + k;
                let right_index = k * 4 + j;
                res[res_index] += left[left_index] * right[right_index];
            }
        }
    }
    res
}

fn mat4_gltf_to_flat(gltf_matrix: [[f32; 4]; 4]) -> [f32; 4 * 4] {
    let mut res = [0.0; 4 * 4];
    for i in 0..4 {
        for j in 0..4 {
            let res_index = j * 4 + i;
            res[res_index] = gltf_matrix[j][i];
        }
    }
    res
}

fn mat4_identity() -> [f32; 4 * 4] {
    [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, //
    ]
}

struct VertexMeshInstance {
    index: usize,
    transform: [f32; 4 * 4],
}

fn load_scene_nodes(gltf_node: &gltf::Node) -> Vec<VertexMeshInstance> {
    log(&format!("Node name = {:?}", gltf_node.name()));
    if let Some(_cam) = gltf_node.camera() {
        log("Node has a camera! Ignored, as cameras are not yet implemented");
    }

    let transform = mat4_gltf_to_flat(gltf_node.transform().matrix());
    let mut nodes = Vec::new();
    for child in gltf_node.children() {
        nodes.extend(load_scene_nodes(&child));
    }

    for node in &mut nodes {
        node.transform = mat4_mul(transform, node.transform);
    }

    if let Some(mesh) = gltf_node.mesh() {
        let index = mesh.index();

        nodes.push(VertexMeshInstance { index, transform });
    }

    nodes
}

#[derive(Debug)]
struct Camera {
    translation: glam::f32::Vec3,
    rotation: glam::f32::Vec4,
    aspect: f32,
    fov: f32,
    far: f32,
    near: f32,
}

fn load_scene_camera(gltf_node: &gltf::Node) -> Option<Camera> {
    let (v_translation, v_rotation, _) = gltf_node.transform().decomposed();
    let translation = glam::f32::Vec3::from(v_translation);
    let rotation = glam::f32::Vec4::from(v_rotation);

    let mut camera = None;
    for child in gltf_node.children() {
        camera = load_scene_camera(&child);
    }

    if let Some(cam) = gltf_node.camera() {
        match cam.projection() {
            gltf::camera::Projection::Perspective(perspective) => {
                camera = Some(Camera {
                    translation: glam::f32::Vec3::ZERO,
                    rotation: glam::f32::Vec4::new(0.0, 0.0, 0.0, 1.0),
                    aspect: perspective.aspect_ratio().unwrap_or(640.0 / 480.0),
                    fov: perspective.yfov(),
                    far: perspective.zfar().unwrap_or(100.0),
                    near: perspective.znear(),
                });
            }

            _ => {}
        }
    }

    camera.map(|c| Camera {
        translation: translation + c.translation,
        rotation: rotation * c.rotation,
        aspect: c.aspect,
        fov: c.fov,
        far: c.far,
        near: c.near,
    })
}

struct VertexMesh {
    bounds: [[f32; 3]; 2],
    vertices: Vec<f32>,
    indices: Vec<u16>,
}

fn load_mesh(gltf_mesh: &Mesh, buffers: &[gltf::buffer::Data]) -> VertexMesh {
    let mut bounds = [[0f32; 3]; 2];
    let mut vertices = Vec::<f32>::new();
    let mut indices = Vec::<u16>::new();

    for prim in gltf_mesh.primitives() {
        log(&format!(
            "\tprimitive = {:?}; bounds = {:?}",
            prim.mode(),
            prim.bounding_box()
        ));

        let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
        if let Some(normal_reader) = reader.read_normals() {
            log(&format!("Model has normals"));
            for (pos, normal) in reader.read_positions().unwrap().zip(normal_reader) {
                vertices.push(pos[0]);
                vertices.push(pos[1]);
                vertices.push(pos[2]);
                vertices.push(normal[0]);
                vertices.push(normal[1]);
                vertices.push(normal[2]);

                bounds[0][0] = bounds[0][0].min(pos[0]);
                bounds[0][1] = bounds[0][1].min(pos[1]);
                bounds[0][2] = bounds[0][2].min(pos[2]);

                bounds[1][0] = bounds[1][0].max(pos[0]);
                bounds[1][1] = bounds[1][1].max(pos[1]);
                bounds[1][2] = bounds[1][2].max(pos[2]);
            }
        } else {
            for pos in reader.read_positions().unwrap() {
                vertices.push(pos[0]);
                vertices.push(pos[1]);
                vertices.push(pos[2]);
                // TODO: Automatically calculate normals if there are none
                vertices.push(0.0);
                vertices.push(0.0);
                vertices.push(0.0);

                bounds[0][0] = bounds[0][0].min(pos[0]);
                bounds[0][1] = bounds[0][1].min(pos[1]);
                bounds[0][2] = bounds[0][2].min(pos[2]);

                bounds[1][0] = bounds[1][0].max(pos[0]);
                bounds[1][1] = bounds[1][1].max(pos[1]);
                bounds[1][2] = bounds[1][2].max(pos[2]);
            }
        }
        for index in reader.read_indices().unwrap().into_u32() {
            indices.push(index as u16);
        }
    }

    VertexMesh {
        bounds,
        vertices,
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
