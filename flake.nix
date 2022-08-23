{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    flake-utils.url = "github:numtide/flake-utils";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };
  };

  outputs = {
    self,
    nixpkgs,
    crane,
    flake-utils,
    rust-overlay,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ (import rust-overlay) ];
      };

      rustWithWasmTarget = pkgs.rust-bin.stable.latest.default.override {
        targets = [ "wasm32-unknown-unknown" ];
      };

      craneLib = (crane.mkLib pkgs).overrideToolchain rustWithWasmTarget;
  
      rust-wasm-gltf = craneLib.buildPackge {
        src = ./.;
        cargExtraArgs = "--target wasm32-unknown-unknown";
        doCheck = false;
      };
    in rec {
      checks = {
        inherit rust-wasm-gltf;
      };

      devShells.default = pkgs.mkShell {
        inputsFrom = builtins.attrValues self.checks;
        nativeBuildInputs = [
          rustWithWasmTarget
          pkgs.nodejs
          pkgs.wasm-pack
          pkgs.pkg-config
          pkgs.openssl
          pkgs.wasm-bindgen-cli
        ];
      };
  });
}
