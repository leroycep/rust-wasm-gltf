import BoomBoxGLB from "./gltf/BoomBox.glb";
import ClearcoatTestGLB from "./gltf/ClearcoatTest/ClearcoatTest.glb";
import FlamingoGLB from "./gltf/Flamingo.glb";
import FlowerGLB from "./gltf/Flower/Flower.glb";
import HorseGLB from "./gltf/Horse.glb";
import IridescenceLampGLB from "./gltf/IridescenceLamp.glb";
import IridescentDishWithOlivesGLB from "./gltf/IridescentDishWithOlives.glb";
import LeePerrySmithGLB from "./gltf/LeePerrySmith/LeePerrySmith.glb";
import LittlestTokyoGLB from "./gltf/LittlestTokyo.glb";
import NefertitiGLB from "./gltf/Nefertiti/Nefertiti.glb";
import ParrotGLB from "./gltf/Parrot.glb";
import PrimaryIonDriveGLB from "./gltf/PrimaryIonDrive.glb";
import RobotExpressiveGLB from "./gltf/RobotExpressive/RobotExpressive.glb";
import ShadowmappableMeshGLB from "./gltf/ShadowmappableMesh.glb";
import SheenChairGLB from "./gltf/SheenChair.glb";
import SoldierGLB from "./gltf/Soldier.glb";
import StorkGLB from "./gltf/Stork.glb";
import XbotGLB from "./gltf/Xbot.glb";
import CoffeematGLB from "./gltf/coffeemat.glb";
import CollisionWorldGLB from "./gltf/collision-world.glb";
import FacecapGLB from "./gltf/facecap.glb";
import FerrariGLB from "./gltf/ferrari.glb";

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
  BoomBoxGLB,
  ClearcoatTestGLB,
  FlamingoGLB,
  FlowerGLB,
  HorseGLB,
  IridescenceLampGLB,
  IridescentDishWithOlivesGLB,
  LeePerrySmithGLB,
  LittlestTokyoGLB,
  NefertitiGLB,
  ParrotGLB,
  PrimaryIonDriveGLB,
  RobotExpressiveGLB,
  ShadowmappableMeshGLB,
  SheenChairGLB,
  SoldierGLB,
  StorkGLB,
  XbotGLB,
  CoffeematGLB,
  CollisionWorldGLB,
  FacecapGLB,
  FerrariGLB,
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