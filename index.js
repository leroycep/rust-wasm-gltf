// For more comments about what's going on here, check out the `hello_world`
// example.
import('./pkg')
  .then(m => {
    console.log("hello, world!");
    console.log(m);
    console.log(m.load_gltf_model);
    const s = m.load_gltf_model("hello, world!");
    console.log(s);
  })
  .catch(console.error);