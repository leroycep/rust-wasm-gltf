// For more comments about what's going on here, check out the `hello_world`
// example.
import('./pkg')
  .then(m => {
    setup_model_buttons(async model_path => {
      // Stop the previous renderer if there is one
      m.quit_rendering();
    
      console.log(`Loading model ${model_path}`);
      const resp = await fetch(model_path, { mode: 'no-cors' });
      if (!resp.ok) {
        throw "failed to download gltf model";
      }
      console.log(resp);
      const gltf_bytes = await resp.arrayBuffer();
      console.log(gltf_bytes);
      quit_prev_renderer = m.load_gltf_model(new Uint8Array(gltf_bytes));
    })
  })
  .catch(console.error);

const models = [
  "/gltf/BoomBox.glb",
  "/gltf/ClearcoatTest/ClearcoatTest.glb",
  "/gltf/Flamingo.glb",
  "/gltf/Flower/Flower.glb",
  "/gltf/Horse.glb",
  "/gltf/IridescenceLamp.glb",
  "/gltf/IridescentDishWithOlives.glb",
  "/gltf/LeePerrySmith/LeePerrySmith.glb",
  "/gltf/LittlestTokyo.glb",
  "/gltf/Nefertiti/Nefertiti.glb",
  "/gltf/Parrot.glb",
  "/gltf/PrimaryIonDrive.glb",
  "/gltf/RobotExpressive/RobotExpressive.glb",
  "/gltf/ShadowmappableMesh.glb",
  "/gltf/SheenChair.glb",
  "/gltf/Soldier.glb",
  "/gltf/Stork.glb",
  "/gltf/Xbot.glb",
  "/gltf/coffeemat.glb",
  "/gltf/collision-world.glb",
  "/gltf/facecap.glb",
  "/gltf/ferrari.glb",
];


function setup_model_buttons(fetch_and_display_model) {
  const buttons_list_element = document.getElementById("models");

  for (let i = 0; i < models.length; i += 1) {
    const button = document.createElement("button");
    
    button.innerText = `Load ${models[i]}`;
    button.addEventListener("click", () => fetch_and_display_model(models[i]));
    
    const li = document.createElement("li");
    li.appendChild(button);
    buttons_list_element.appendChild(li);
  }
}