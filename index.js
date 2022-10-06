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
    const load_model_at_path = async model_path => {
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
      m.load_gltf_model(new Uint8Array(gltf_bytes));
    };
    setup_model_buttons(load_model_at_path);
    load_model_at_path(SoldierGLB);
  })
  .catch(console.error);

const models = [
  { name: "Flower", path: FlowerGLB },
  { name: "Stork", path: StorkGLB },
  { name: "Flamingo", path: FlamingoGLB },
  { name: "CollisionWorld", path: CollisionWorldGLB },
  { name: "Parrot", path: ParrotGLB },
  { name: "Horse", path: HorseGLB },
  { name: "ClearcoatTest", path: ClearcoatTestGLB },
  { name: "Facecap", path: FacecapGLB },
  { name: "ShadowmappableMesh", path: ShadowmappableMeshGLB },
  { name: "LeePerrySmith", path: LeePerrySmithGLB },
  { name: "RobotExpressive", path: RobotExpressiveGLB },
  { name: "Nefertiti", path: NefertitiGLB },
  { name: "Ferrari", path: FerrariGLB },
  { name: "Soldier", path: SoldierGLB },
  { name: "Xbot", path: XbotGLB },
  { name: "IridescentDishWithOlives", path: IridescentDishWithOlivesGLB },
  { name: "LittlestTokyo", path: LittlestTokyoGLB },
  { name: "SheenChair", path: SheenChairGLB },
  { name: "Coffeemat", path: CoffeematGLB },
  { name: "IridescenceLamp", path: IridescenceLampGLB },
  { name: "PrimaryIonDrive", path: PrimaryIonDriveGLB },
  { name: "BoomBox", path: BoomBoxGLB },
];


function setup_model_buttons(fetch_and_display_model) {
  const buttons_list_element = document.getElementById("models");

  for (let i = 0; i < models.length; i += 1) {
    const button = document.createElement("button");
    
    button.innerText = `Load ${models[i].name}`;
    button.addEventListener("click", () => fetch_and_display_model(models[i].path));
    
    const li = document.createElement("li");
    li.appendChild(button);
    buttons_list_element.appendChild(li);
  }
}