# Python environment nix shell file for reproducibility

with import <nixpkgs> { };

let
    pythonPackages = python3Packages;
in pkgs.mkShell rec {
    name = "impurePythonEnv";
    venvDir = "./.venv";
    buildInputs = [
        pythonPackages.python
        pythonPackages.venvShellHook
        pythonPackages.numpy
        pythonPackages.requests

        git
    ];

    postVenvCreation = ''
        unset SOURCE_DATE_EPOCH
        pip install -r requirements.txt
    '';

    postShellHook = ''
        unset SOURCE_DATE_EPOCH
    '';
}

