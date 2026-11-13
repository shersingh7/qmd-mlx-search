{
  description = "QMD - Quick Markdown Search";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      # These are the native hosts exercised by .github/workflows/nix.yml.
      # Adding another host requires its own verified dependency hash and smoke.
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];
    in
    {
      homeModules.default = { config, lib, pkgs, ... }:
        with lib;
        let
          cfg = config.programs.qmd;
        in
        {
          options.programs.qmd = {
            enable = mkEnableOption "QMD - on-device search engine for markdown notes";

            package = mkOption {
              type = types.package;
              default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
              defaultText = literalExpression "inputs.qmd.packages.\${pkgs.stdenv.hostPlatform.system}.default";
              description = "The qmd package to use.";
            };
          };

          config = mkIf cfg.enable {
            home.packages = [ cfg.package ];
          };
        };
    } //
    flake-utils.lib.eachSystem supportedSystems (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        packageJson = builtins.fromJSON (builtins.readFile ./package.json);
        version = packageJson.version;
        # One exact Node derivation supplies the runtime AND native addon headers.
        # The HTTP transport requires Node >=22; Bun is only the locked fetcher.
        nodejs = pkgs.nodejs_22;
        # nixpkgs' wrapper hardcodes its own Node and npm_config_nodedir. Override
        # both: a CLI --nodedir alone still compiled the addon against Node 24.
        nodeGyp = (pkgs.node-gyp.override { inherit nodejs; }).overrideAttrs (_old: {
          makeWrapperArgs = [ "--set npm_config_nodedir ${pkgs.lib.getDev nodejs}" ];
        });

        source = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter = path: type:
            let
              name = builtins.baseNameOf path;
              relative = pkgs.lib.removePrefix "${toString ./.}/" (toString path);
              top = builtins.head (pkgs.lib.splitString "/" relative);
            in
            # Only build/package inputs enter the store; never local indexes,
            # models, finetuning data or unrelated checkout artifacts.
            (toString path == toString ./. || builtins.elem top [
              "src" "scripts" "bin" "package.json" "tsconfig.json"
              "tsconfig.build.json" "LICENSE" "CHANGELOG.md"
            ]) && !(builtins.elem name [
              ".git" "node_modules" "dist" ".venv" "__pycache__"
              ".pytest_cache" ".direnv" "result"
            ]) && !(pkgs.lib.hasSuffix ".pyc" name || pkgs.lib.hasSuffix ".pyo" name);
        };

        nodeModulesHashes = {
          # Recompute both from native CI after package.json/bun.lock or the
          # build+production dependency layout changes; never disable integrity.
          # Linux measured by run 43452002860 at 093c9f7.
          x86_64-linux = "sha256-lYii6J60m9lQmhDmCjP4lYmEdDxkExke+XIrRN7Vuho=";
          # Darwin measured by attempt 2 of the same exact-head run.
          aarch64-darwin = "sha256-bF4vRAwHesw3dFXS7q4sFyRxFmgFpiUdfv/8TTOyBuY=";
        };

        nodeModules = pkgs.stdenvNoCC.mkDerivation {
          pname = "qmd-node-modules";
          inherit version;

          # No checkout .npmrc/bunfig, prepare hook source, caches or modules can
          # contaminate the fixed-output fetch. Integrity verification stays on.
          src = pkgs.lib.fileset.toSource {
            root = ./.;
            fileset = pkgs.lib.fileset.unions [ ./package.json ./bun.lock ];
          };

          impureEnvVars = pkgs.lib.fetchers.proxyImpureEnvVars;
          nativeBuildInputs = [ pkgs.bun ];
          dontConfigure = true;

          buildPhase = ''
            runHook preBuild
            export HOME="$TMPDIR/qmd-fetch-home"
            mkdir -p "$HOME"

            # Build with the lock's compiler/types, but ship production deps only.
            # Neither pass may run dependency or root lifecycle scripts.
            bun install --backend copyfile --frozen-lockfile --ignore-scripts \
              --no-progress --cache-dir "$TMPDIR/bun-cache"
            mv node_modules node_modules-build
            bun install --backend copyfile --frozen-lockfile --ignore-scripts \
              --no-progress --cache-dir "$TMPDIR/bun-cache" --production
            runHook postBuild
          '';

          installPhase = ''
            runHook preInstall
            mkdir -p $out
            cp -R node_modules node_modules-build $out/
            runHook postInstall
          '';

          dontFixup = true;
          outputHash = nodeModulesHashes.${system};
          outputHashAlgo = "sha256";
          outputHashMode = "recursive";
        };

        qmd = pkgs.stdenv.mkDerivation {
          pname = "qmd";
          inherit version;
          src = source;
          dontConfigure = true;

          nativeBuildInputs = [
            nodejs
            pkgs.makeWrapper
            nodeGyp
            pkgs.python3
          ] ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
            pkgs.darwin.cctools
          ];

          buildPhase = ''
            runHook preBuild
            export HOME="$TMPDIR/qmd-build-home"
            export PYTHON=${pkgs.python3}/bin/python3
            mkdir -p "$HOME"

            cp -R ${nodeModules}/node_modules-build node_modules
            chmod -R u+w node_modules
            ${nodejs}/bin/node scripts/build.mjs

            rm -rf node_modules
            cp -R ${nodeModules}/node_modules ./
            chmod -R u+w node_modules
            # Offline source build of exactly the locked better-sqlite3. Do not
            # download prebuilds/headers or use node-gyp's runtime ABI implicitly.
            (cd node_modules/better-sqlite3 && \
              node-gyp rebuild --release --nodedir=${pkgs.lib.getDev nodejs})
            runHook postBuild
          '';

          installPhase = ''
            runHook preInstall
            mkdir -p $out/lib/qmd $out/bin
            cp -R node_modules $out/lib/qmd/
            cp -R dist scripts bin $out/lib/qmd/
            cp package.json LICENSE CHANGELOG.md $out/lib/qmd/

            # Do not consult a consumer's Bun/Node selection for a Node-ABI addon.
            # better-sqlite3's bundled SQLite loads sqlite-vec without loader-path hacks.
            makeWrapper ${nodejs}/bin/node $out/bin/qmd \
              --add-flags "$out/lib/qmd/dist/cli/qmd.js" \
              --prefix PATH : ${pkgs.lib.makeBinPath [ nodejs ]}
            runHook postInstall
          '';

          doInstallCheck = true;
          installCheckPhase = ''
            runHook preInstallCheck
            export HOME="$TMPDIR/qmd-install-home"
            export XDG_CONFIG_HOME="$HOME/config"
            export XDG_CACHE_HOME="$HOME/cache"
            mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
            test -s $out/lib/qmd/dist/index.js
            test -s $out/lib/qmd/dist/index.d.ts
            test -s $out/lib/qmd/scripts/qmd_mlx/server.py
            $out/bin/qmd --help > "$TMPDIR/qmd-help.txt"
            ${nodejs}/bin/node --input-type=module -e "
              const {openDatabase,loadSqliteVec}=await import('file://$out/lib/qmd/dist/db.js');
              const db=openDatabase(':memory:');
              try {
                loadSqliteVec(db);
                if (!db.prepare('select vec_version() as version').get().version) throw Error('sqlite-vec unavailable');
              } finally { db.close(); }
            "
            runHook postInstallCheck
          '';

          passthru = { inherit nodejs nodeModules; };
          meta = with pkgs.lib; {
            description = "On-device search engine for markdown notes and knowledge bases";
            homepage = "https://github.com/shersingh7/qmd-mlx-search";
            license = licenses.mit;
            mainProgram = "qmd";
            platforms = supportedSystems;
          };
        };

        installedCheck = pkgs.writeShellApplication {
          name = "qmd-check-installed";
          runtimeInputs = [ nodejs pkgs.python3 ];
          text = ''
            if [ "$#" -lt 1 ]; then
              echo "usage: qmd-check-installed /path/to/nix-output [--receipt /new/file.json]" >&2
              exit 2
            fi
            package="$1"
            shift
            : "''${TMPDIR:?Set TMPDIR to a managed scratch directory}"
            exec ${pkgs.python3}/bin/python3 ${./test/security/test_nix_packaging.py} \
              --installed "$package" --node ${nodejs}/bin/node "$@"
          '';
        };
      in
      {
        packages = {
          default = qmd;
          inherit qmd;
        };

        apps = {
          default = {
            type = "app";
            program = "${qmd}/bin/qmd";
          };
          # Run outside the build sandbox so loopback HTTP acceptance is not
          # confused with sandbox socket policy. The app pins the same Node ABI.
          check-installed = {
            type = "app";
            program = "${installedCheck}/bin/qmd-check-installed";
          };
        };

        # The Python/MLX helpers are shipped recursively as source/assets, not a
        # bundled GPU/MLX environment. Provision their pinned requirements in a
        # separate compatible environment; these checks never load any model.
        devShells.default = pkgs.mkShell {
          packages = [ nodejs pkgs.bun pkgs.python3 ];
          shellHook = ''
            echo "QMD development shell (Node.js ${nodejs.version})"
            echo "Run: npm run build; node dist/cli/qmd.js --help"
          '';
        };
      }
    );
}
