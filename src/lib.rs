use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{WebGlProgram, WebGlRenderingContext, WebGlShader};

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into::<web_sys::HtmlCanvasElement>()?;

    let context = canvas
        .get_context("webgl")?
        .unwrap()
        .dyn_into::<WebGlRenderingContext>()?;

    let vert_shader = compile_shader(
        &context,
        WebGlRenderingContext::VERTEX_SHADER,
        r#"
        attribute vec4 position;
        void main() {
            gl_Position = position;
        }
    "#,
    )?;
    let frag_shader = compile_shader(
        &context,
        WebGlRenderingContext::FRAGMENT_SHADER,
        r#"
        void main() {
            gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
    "#,
    )?;
    let program = link_program(&context, &vert_shader, &frag_shader)?;
    context.use_program(Some(&program));

    let vertices: [f32; 9] = [-0.7, -0.7, 0.0, 0.7, -0.7, 0.0, 0.0, 0.7, 0.0];

    let buffer = context.create_buffer().ok_or("failed to create buffer")?;
    context.bind_buffer(WebGlRenderingContext::ARRAY_BUFFER, Some(&buffer));

    // Note that `Float32Array::view` is somewhat dangerous (hence the
    // `unsafe`!). This is creating a raw view into our module's
    // `WebAssembly.Memory` buffer, but if we allocate more pages for ourself
    // (aka do a memory allocation in Rust) it'll cause the buffer to change,
    // causing the `Float32Array` to be invalid.
    //
    // As a result, after `Float32Array::view` we have to be very careful not to
    // do any memory allocations before it's dropped.
    unsafe {
        let vert_array = js_sys::Float32Array::view(&vertices);

        context.buffer_data_with_array_buffer_view(
            WebGlRenderingContext::ARRAY_BUFFER,
            &vert_array,
            WebGlRenderingContext::STATIC_DRAW,
        );
    }

    context.vertex_attrib_pointer_with_i32(0, 3, WebGlRenderingContext::FLOAT, false, 0, 0);
    context.enable_vertex_attrib_array(0);

    context.clear_color(0.0, 0.0, 0.0, 1.0);
    context.clear(WebGlRenderingContext::COLOR_BUFFER_BIT);

    context.draw_arrays(
        WebGlRenderingContext::TRIANGLES,
        0,
        (vertices.len() / 3) as i32,
    );
    Ok(())
}

pub fn compile_shader(
    context: &WebGlRenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, String> {
    let shader = context
        .create_shader(shader_type)
        .ok_or_else(|| String::from("Unable to create shader object"))?;
    context.shader_source(&shader, source);
    context.compile_shader(&shader);

    if context
        .get_shader_parameter(&shader, WebGlRenderingContext::COMPILE_STATUS)
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
    context: &WebGlRenderingContext,
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
        .get_program_parameter(&program, WebGlRenderingContext::LINK_STATUS)
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
    let (document, buffers, _images) =
        gltf::import_slice(&gltf_bytes).map_err(|err| format!("error parsing gltf: {}", err))?;

    let mut bounds = [[0f32; 3]; 2];
    let mut vertices = Vec::<f32>::new();
    let mut indices = Vec::<u16>::new();

    for mesh in document.meshes() {
        for prim in mesh.primitives() {
            log(&format!(
                "\tprimitive = {:?}; bounds = {:?}",
                prim.mode(),
                prim.bounding_box()
            ));

            let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
            for pos in reader.read_positions().unwrap() {
                vertices.push(pos[0]);
                vertices.push(pos[1]);
                vertices.push(pos[2]);

                bounds[0][0] = bounds[0][0].min(pos[0]);
                bounds[0][1] = bounds[0][1].min(pos[1]);
                bounds[0][2] = bounds[0][2].min(pos[2]);

                bounds[1][0] = bounds[1][0].max(pos[0]);
                bounds[1][1] = bounds[1][1].max(pos[1]);
                bounds[1][2] = bounds[1][2].max(pos[2]);
            }
            for index in reader.read_indices().unwrap().into_u32() {
                indices.push(index as u16);
            }
        }
    }

    log(&format!(
        "Loaded {} vertices, {} indices; bounds = {:#?}",
        vertices.len() / 3,
        indices.len(),
        bounds,
    ));

    display_model(bounds, &vertices, &indices)?;

    Ok(())
}

pub fn display_model(
    bounds: [[f32; 3]; 2],
    vertices: &[f32],
    indices: &[u16],
) -> Result<(), JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into::<web_sys::HtmlCanvasElement>()?;

    let context = canvas
        .get_context("webgl")?
        .unwrap()
        .dyn_into::<WebGlRenderingContext>()?;

    let vert_shader = compile_shader(
        &context,
        WebGlRenderingContext::VERTEX_SHADER,
        r#"
        uniform mat4 transform;
        attribute vec3 position;
        void main() {
            gl_Position = transform * vec4(position, 1.0);
        }
    "#,
    )?;
    let frag_shader = compile_shader(
        &context,
        WebGlRenderingContext::FRAGMENT_SHADER,
        r#"
        void main() {
            gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
    "#,
    )?;
    let program = link_program(&context, &vert_shader, &frag_shader)?;
    context.use_program(Some(&program));

    let transform_uniform = context
        .get_uniform_location(&program, "transform")
        .ok_or_else(|| JsValue::from_str("Could not get transform uniform location"))?;

    let buffer = context.create_buffer().ok_or("failed to create buffer")?;
    context.bind_buffer(WebGlRenderingContext::ARRAY_BUFFER, Some(&buffer));

    // Note that `Float32Array::view` is somewhat dangerous (hence the
    // `unsafe`!). This is creating a raw view into our module's
    // `WebAssembly.Memory` buffer, but if we allocate more pages for ourself
    // (aka do a memory allocation in Rust) it'll cause the buffer to change,
    // causing the `Float32Array` to be invalid.
    //
    // As a result, after `Float32Array::view` we have to be very careful not to
    // do any memory allocations before it's dropped.
    unsafe {
        let vert_array = js_sys::Float32Array::view(vertices);

        context.buffer_data_with_array_buffer_view(
            WebGlRenderingContext::ARRAY_BUFFER,
            &vert_array,
            WebGlRenderingContext::STATIC_DRAW,
        );
    }

    let index_buffer = context
        .create_buffer()
        .ok_or("failed to create index buffer")?;
    context.bind_buffer(
        WebGlRenderingContext::ELEMENT_ARRAY_BUFFER,
        Some(&index_buffer),
    );

    unsafe {
        let index_array = js_sys::Uint16Array::view(indices);

        context.buffer_data_with_array_buffer_view(
            WebGlRenderingContext::ELEMENT_ARRAY_BUFFER,
            &index_array,
            WebGlRenderingContext::STATIC_DRAW,
        );
    }

    context.vertex_attrib_pointer_with_i32(0, 3, WebGlRenderingContext::FLOAT, false, 0, 0);
    context.enable_vertex_attrib_array(0);

    let num_indices = (indices.len()) as i32;

    let f = Rc::new(RefCell::new(None));
    let g = f.clone();

    let mut i: u64 = 0;
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        context.clear_color(0.0, 0.0, 0.0, 1.0);
        context.clear(WebGlRenderingContext::COLOR_BUFFER_BIT);

        let theta = (i as f32) / 60.0;
        let tcos = theta.cos();
        let tsin = theta.sin();

        let transform = mat4_mul(
            // Scale down model to fit in viewport
            [
                0.01, 0.0, 0.0, 0.0, //
                0.0, 0.01, 0.0, 0.0, //
                0.0, 0.0, 0.01, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
            ],
            // Rotate the model
            [
                tcos, 0.0, tsin, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                -tsin, 0.0, tcos, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
            ],
        );

        context.uniform_matrix4fv_with_f32_array(Some(&transform_uniform), false, &transform);

        context.draw_elements_with_i32(
            WebGlRenderingContext::TRIANGLES,
            num_indices,
            WebGlRenderingContext::UNSIGNED_SHORT,
            0,
        );

        i += 1;
        request_animation_frame(f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
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
