# Python environment nix shell file for reproducibility

with import <nixpkgs> { };

let
    pythonPackages = python3Packages;
in pkgs.mkShell rec {
    name = "Python-R-env";
    venvDir = "./.venv";
    buildInputs = [
        pythonPackages.python
        pythonPackages.venvShellHook
        pythonPackages.numpy
        pythonPackages.requests

        R
        rPackages.rmarkdown
        rPackages.knitr
        rPackages.tidyverse
        rPackages.plotly
        rPackages.gapminder
        rPackages.reticulate
        rPackages.dslabs
        rPackages.patchwork
        rPackages.Rtsne

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

