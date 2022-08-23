// For more comments about what's going on here, check out the `hello_world`
// example.
import('./pkg')
  .then(m => {
    fetch("/gltf/Stork.glb", { mode: 'no-cors' })
      .then(resp => {
        console.log(resp);
        if (!resp.ok) {
          throw "failed to download gltf model";
        }
        return resp.arrayBuffer();
      })
      .then(gltf_bytes => {
        console.log(gltf_bytes);
        console.log(m.load_gltf_model(new Uint8Array(gltf_bytes)));
      })
  })
  .catch(console.error);