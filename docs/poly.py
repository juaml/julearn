"""Configure sphinx-polyversion for building documentation."""

import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader
from sphinx_polyversion import apply_overrides, logger
from sphinx_polyversion.git import (
    Git,
    GitRefType,
    file_predicate,
    refs_by_type,
)
from sphinx_polyversion.pyvenv import Pip
from sphinx_polyversion.setuptools_scm import SetuptoolsScmDriver
from sphinx_polyversion.sphinx import SphinxBuilder


#: Regex matching the branches to build docs for
BRANCH_REGEX = r"main"

#: Regex matching the tags to build docs for
TAG_REGEX = r"v\d+\.\d+\.\d+"

#: Output dir relative to project root
#: !!! This name has to be choosen !!!
OUTPUT_DIR = "docs/_build"

#: Source directory relative to project root
SOURCE_DIR = "./"

#: Arguments to pass to `pip install`
PIP_ARGS = "-e .[docs,viz]"

#: Arguments to pass to `sphinx-build`
SPHINX_ARGS = "-E -v"

#: Whether to build using only local files and mock data
MOCK = False

#: Whether to run the builds in sequence or in parallel
SEQUENTIAL = False

#: Whether to build all versions or only the missing ones (if False, only
# versions that are not already built will be built)
BUILD_ALL = False

# Whether to skip the HTML patching step (for doc development purposes,
# since patching is slow)
NO_PATCH = False

# Load overrides read from commandline to global scope
apply_overrides(globals())

# Determine repository root directory
root = Git.root(Path(__file__).parent)

logger.info(f"Building documentation for repository at {root}")

# Setup driver and run it
docs_src = Path(SOURCE_DIR) / "docs"  # convert from string
out_dir = root / OUTPUT_DIR

logger.info(f"Docs Source directory: {docs_src}")
logger.info(f"Output directory: {out_dir}")

# Builders by version:
BUILDER = {
    None: SphinxBuilder(docs_src, args=SPHINX_ARGS.split()),  # default
}


class PipSetupToolsSCM(Pip):
    """Custom pip environment that sets the version for setuptools_scm."""

    # Somehow, setuptools_scm doesn't work when we pretend the version
    # specifically for julearn, so we need to override the version for all
    # environments. This is a bit hacky, but it works.
    def apply_overrides(self, env: dict[str, str]) -> dict[str, str]:
        """Apply overrides to the environment variables.

        Args:
            env: The environment variables to apply the overrides to.

        Returns:
            The environment variables with the overrides applied.

        """
        self.env["SETUPTOOLS_SCM_PRETEND_VERSION"] = self.env[
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_JULEARN"
        ]
        logger.info(f"Applying overrides: {self.env}")
        return super().apply_overrides(env)


# Custom UV environment creator
class NewVenvUVEnvCreator:
    """Python environment using uv as the package manager."""

    def __init__(
        self,
        python_version: str | None = None,
    ):
        """Initialize UV environment provider.

        Args:
            args: Arguments to pass to `uv pip install`
            env: Environment variables to set
            python_version: Python version to use. If none, use the default.

        """
        self.python_version = python_version

    async def __call__(self, path: Path) -> None:
        """Create the virtual python environment.

        Args:
            path: Path to the virtual environment directory

        """

        # Create virtual environment
        uvvenvpath = path.parent / ".uvenv"
        subprocess.check_call([sys.executable, "-m", "venv", str(uvvenvpath)])

        # Get the pip path inside the venv
        if os.name == "nt":  # Windows
            pip_path = uvvenvpath / "Scripts" / "pip"
            python_path = uvvenvpath / "Scripts" / "python"
        else:  # Unix/macOS
            pip_path = uvvenvpath / "bin" / "pip"
            python_path = uvvenvpath / "bin" / "python"

        # Install uv in the virtual environment
        subprocess.check_call([str(pip_path), "install", "uv"])

        # Get the uv path inside the venv
        if os.name == "nt":  # Windows
            uv_path = uvvenvpath / "Scripts" / "uv"
        else:  # Unix/macOS
            uv_path = uvvenvpath / "bin" / "uv"

        # now use uv to create the environment with the correct python
        cmd = [str(uv_path), "venv", str(path)]
        if self.python_version is not None:
            cmd += ["--python", self.python_version]
        subprocess.check_call(cmd)

        if os.name == "nt":  # Windows
            python_path = path / "Scripts" / "python"
        else:  # Unix/macOS
            python_path = path / "bin" / "python"

        # Now install pip in the environment using uv
        cmd = [
            str(uv_path),
            "pip",
            "install",
            "--python",
            str(python_path),
            "pip",
        ]
        subprocess.check_call(cmd, cwd=path)


ENVIRONMENT = {
    None: PipSetupToolsSCM.factory(
        venv=Path(".venv"),
        args=PIP_ARGS.split(),
        creator=NewVenvUVEnvCreator(),
        temporary=True,
    ),  # default
    r"v0\.2\.\d+": PipSetupToolsSCM.factory(
        venv=Path(".venv"),
        args=(
            "-e . -r docs-requirements.txt seaborn==0.11.1 numpy==1.19.3 "
            "scikit-learn==0.23.2"
        ).split(),
        creator=NewVenvUVEnvCreator(python_version="3.8"),
        temporary=True,
    ),  # for older versions, install without docs extras
    r"v0.2.5": PipSetupToolsSCM.factory(
        venv=Path(".venv"),
        args=(
            "-e . -r docs-requirements.txt numpy==1.23.1 scikit-learn==1.0.2"
        ).split(),
        creator=NewVenvUVEnvCreator(python_version="3.8"),
        temporary=True,
    ),
    r"v0.2.7": PipSetupToolsSCM.factory(
        venv=Path(".venv"),
        args=(
            "-e . -r docs-requirements.txt numpy==1.23.5 scikit-learn==1.0.2"
        ).split(),
        creator=NewVenvUVEnvCreator(python_version="3.8"),
        temporary=True,
    ),
    r"v0.3.0": PipSetupToolsSCM.factory(
        venv=Path(".venv"),
        args=(
            "-e .[docs,viz] bokeh==3.1.0 panel==1.0.0b1 "
            "param==1.12.0 scikit-learn==1.3.0"
        ).split(),
        creator=NewVenvUVEnvCreator(python_version="3.11"),
        temporary=True,
    ),
    r"v0\.3\.\d+": PipSetupToolsSCM.factory(
        venv=Path(".venv"),
        args="-e .[docs,viz] panel==1.3.0 param==2.0.0".split(),
        creator=NewVenvUVEnvCreator(python_version="3.11"),
        temporary=True,
    ),
    r"v0.3.5": PipSetupToolsSCM.factory(
        venv=Path(".venv"),
        args="-e .[docs,viz]".split(),
        creator=NewVenvUVEnvCreator(python_version="3.11"),
        temporary=True,
    ),
}

template_dir = root / docs_src / "poly_files" / "templates"
static_dir = root / docs_src / "poly_files" / "static"
patches_dir = root / docs_src / "poly_files" / "patches"

logger.info(f"Template directory: {template_dir}")
logger.info(f"Static directory: {static_dir}")
logger.info(f"Patches directory: {patches_dir}")


def closest_tag(ref, tags) -> "str | None":
    """Find the closest tag for a given version.

    Args:
        ref: Git reference
        tags: Available tags in the environment mapping

    Returns:
        The closest tag name that should be used to select the environment

    """

    logger.info(f"Root: {root}, Ref: {ref}, Tags: {tags}")
    this_tag = ref.name
    out = None
    if this_tag in tags:
        logger.info(f"Exact match found for tag: {this_tag}")
        out = this_tag
    else:
        for tag in tags:
            logger.info(f"Checking if {this_tag} matches regexp {tag}...")
            if tag is None:
                continue
            if re.match(re.compile(tag), this_tag):
                logger.info(
                    f"Prefix match found: {this_tag} starts with {tag}"
                )
                out = tag
                break
    logger.info(f"Closest tag: {out}")
    return out


# Now check which versions polyversion will run for
vsc = Git(
    branch_regex=BRANCH_REGEX,
    tag_regex=TAG_REGEX,
    predicate=file_predicate([docs_src]),  # exclude refs without source dir
)

refs = asyncio.run(vsc.retrieve(root))
logger.info(f"Retrieved refs: {[ref.name for ref in refs]}")


sorted_tags = sorted(
    [ref for ref in refs if ref.type_ == GitRefType.TAG], key=lambda x: x.name
)
if len(sorted_tags) == 0:
    logger.warning(
        "No tags found matching the regex. Latest tag will be None."
    )
    latest_tag = None
else:
    latest_tag = sorted_tags[-1]
    logger.info(f"Latest tag: {latest_tag.name}")

if not BUILD_ALL:
    to_build = []

    for ref in refs:
        if ref.type_ != GitRefType.TAG:
            continue
        if (out_dir / ref.name / "index.html").exists():
            logger.info(
                f"Documentation for {ref.name} already exists, skipping build."
            )
        else:
            logger.info(
                f"Documentation for {ref.name} does not exist, "
                "adding to build list."
            )
            to_build.append(ref.name)

    TAG_REGEX = (
        "|".join(to_build) if to_build else r"^$"
    )  # match nothing if empty
    logger.info(f"Updated TAG_REGEX to: {TAG_REGEX}")


#: Data passed to templates
def data(driver, rev, env):
    """Create a factory for data passed to templates.

    Args:
        driver: The driver instance running the build
        rev: The current revision being built
        env: The environment variables for the current build

    """
    revisions = driver.targets
    branches, tags = refs_by_type(revisions)
    latest = latest_tag
    return {
        "current": rev,
        "tags": tags,
        "branches": branches,
        "revisions": revisions,
        "latest": latest,
    }


def root_data(driver):
    """Create a factory for root data passed to templates.

    Args:
        driver: The driver instance running the build

    """
    revisions = driver.builds
    latest = latest_tag
    return {"revisions": revisions, "latest": latest}


SetuptoolsScmDriver(
    root,
    out_dir,
    vcs=Git(
        branch_regex=BRANCH_REGEX,
        tag_regex=TAG_REGEX,
        predicate=file_predicate(
            [docs_src]
        ),  # exclude refs without source dir
    ),
    builder=BUILDER,
    env=ENVIRONMENT,
    selector=closest_tag,
    template_dir=template_dir,
    # static_dir=static_dir,
    data_factory=data,
    root_data_factory=root_data,
).run()

if NO_PATCH:
    logger.info("NO_PATCH is set, skipping HTML patching.")
    sys.exit(0)

# Patch all HTML files with the new version of the selector that includes
# the version selector. This is necessary because the selector is included as a
# static file and is not rendered by jinja. We can do this by using
# BeautifulSoup to parse the HTML files and replace the selector with the new
# one that includes the version selector. This is a bit hacky, but it works.

logger.info("Patching HTML files with new version selector...")

# Check all of the built directories to find built versions
built_versions = {
    "branches": [],
    "tags": [],
}
for ref in refs:
    if (out_dir / ref.name).is_dir():
        if ref.type_ == GitRefType.BRANCH:
            built_versions["branches"].append(ref)
        elif ref.type_ == GitRefType.TAG:
            built_versions["tags"].append(ref)


env = Environment(loader=FileSystemLoader(patches_dir))
version_template = env.get_template("versions.html")
versions_rtd_template = env.get_template("versions_rtd.html")
version_banner_template = env.get_template("version_banner.html")
version_banner_rtd_template = env.get_template("version_banner_rtd.html")
new_css = root / docs_src / "_static" / "css" / "version_selector.css"


def _version_root(html_file: Path, directory: Path) -> str:
    """Return the relative prefix needed to reach the root from html_file.

    For a file N levels deep inside `directory`, the versions root (parent of
    `directory`) requires N+1 `../` steps.
    """
    depth = len(html_file.relative_to(directory).parts) - 1  # -1 for filename
    return "../" * (depth + 1)


def recursive_patch_html_files(directory: Path, html: str, div_selector: str):
    """Recursively patch HTML files in a directory with the given HTML.

    Args:
        directory: The directory to recursively patch HTML files in.
        html: The HTML string to insert into the files.
        div_selector: The class name of the div to insert the HTML after if
            the version selector div is not found.

    """
    for html_file in directory.glob("**/*.html"):
        file_html = html.replace(
            "{VERSIONROOT}", _version_root(html_file, directory)
        )
        with open(html_file, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        # Look for furo-native placeholder first, fall back to RTD-style div
        selector_div = soup.find(
            "div", {"id": "version-selector"}
        ) or soup.find("div", {"class": "rst-versions"})
        if selector_div is not None:
            selector_div.replace_with(BeautifulSoup(file_html, "html.parser"))
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(str(soup))
        else:
            # div was not there, so we need to append it after
            nav_div = soup.find("div", {"class": div_selector})
            if nav_div is not None:
                nav_div.insert_after(BeautifulSoup(file_html, "html.parser"))
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(str(soup))
            else:
                logger.warning(
                    "Could not find nav div to insert version selector for "
                    f"{html_file}"
                )


def recursive_patch_html_files_add_banner(directory: Path, banner_html: str):
    """Inject a version banner into the toc-drawer aside (furo theme).

    If the aside has the ``no-toc`` class (meaning the TOC is hidden), that
    class is removed so the drawer is visible and the banner is shown.
    """
    for html_file in directory.glob("**/*.html"):
        file_banner = banner_html.replace(
            "{VERSIONROOT}", _version_root(html_file, directory)
        )
        with open(html_file, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        toc_drawer = soup.find("aside", class_="toc-drawer")
        if toc_drawer is not None:
            # Activate the drawer if it was hidden (no-toc pages)
            classes = toc_drawer.get("class", [])
            if "no-toc" in classes:
                classes.remove("no-toc")
                toc_drawer["class"] = classes
                # Also remove no-toc from the header and content toggle labels
                for label in soup.find_all("label", class_="toc-overlay-icon"):
                    label_classes = label.get("class", [])
                    if "no-toc" in label_classes:
                        label_classes.remove("no-toc")
                        label["class"] = label_classes
            # Remove any existing banner before inserting
            existing = toc_drawer.find("div", {"class": "version-banner"})
            if existing is not None:
                existing.decompose()
            toc_drawer.insert(0, BeautifulSoup(file_banner, "html.parser"))
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(str(soup))
        else:
            logger.warning(
                "Could not find toc-drawer to insert version banner for "
                f"{html_file}"
            )


def recursive_patch_html_files_add_banner_rtd(
    directory: Path, banner_html: str
):
    """Inject a version banner at the top of the RTD theme main content div."""
    for html_file in directory.glob("**/*.html"):
        file_banner = banner_html.replace(
            "{VERSIONROOT}", _version_root(html_file, directory)
        )
        with open(html_file, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        main_div = soup.find("div", {"role": "main", "class": "document"})
        if main_div is not None:
            # Remove any existing banner before inserting
            existing = main_div.find(
                "div", style=lambda s: s and "border-left" in s
            )
            if existing is not None:
                existing.decompose()
            main_div.insert(0, BeautifulSoup(file_banner, "html.parser"))
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(str(soup))
        else:
            logger.warning(
                "Could not find RTD main div to insert version banner for "
                f"{html_file}"
            )


def ensure_css_link(directory: Path):
    """Ensure version_selector.css link is present."""
    for html_file in directory.glob("**/*.html"):
        depth = len(html_file.relative_to(directory).parts) - 1
        css_href = "../" * depth + "_static/css/version_selector.css"
        with open(html_file, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        head = soup.find("head")
        if head is None:
            continue
        # Remove any stale version_selector link (wrong depth) before adding
        # the correct one
        for old_link in soup.find_all(
            "link", href=lambda h: h and "version_selector.css" in h
        ):
            old_link.decompose()
        new_link = soup.new_tag("link", rel="stylesheet", href=css_href)
        head.append(new_link)
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(str(soup))


for built_dir in out_dir.iterdir():
    if built_dir.is_dir():
        if built_dir.name.startswith("v"):
            logger.info(f"Patching built version directory: {built_dir.name}")
            current_version = next(
                [
                    tag
                    for tag in built_versions["tags"]
                    if tag.name == built_dir.name
                ]
            )
            if int(built_dir.name.split(".")[1]) < 3:
                # Old read_the_docs template
                version_rtd_html = versions_rtd_template.render(
                    versions=built_versions,
                    current_version=current_version,
                )
                banner_rtd_html = version_banner_rtd_template.render(
                    current_version=current_version,
                    latest=latest_tag,
                )
                recursive_patch_html_files(
                    built_dir, version_rtd_html, "wy-grid-for-nav"
                )
                recursive_patch_html_files_add_banner_rtd(
                    built_dir, banner_rtd_html
                )
            else:
                version_html = version_template.render(
                    versions=built_versions,
                    current_version=current_version,
                    latest=latest_tag,
                )
                banner_html = version_banner_template.render(
                    current_version=current_version,
                    latest=latest_tag,
                )
                recursive_patch_html_files(
                    built_dir, version_html, "sidebar-tree"
                )
                recursive_patch_html_files_add_banner(built_dir, banner_html)

                # Copy CSS and ensure link is present for pre-existing builds
                static_dir = built_dir / "_static" / "css"
                static_dir.mkdir(parents=True, exist_ok=True)
                with open(new_css, encoding="utf-8") as f:
                    new_css_content = f.read()
                with open(
                    static_dir / "version_selector.css", "w", encoding="utf-8"
                ) as f:
                    f.write(new_css_content)
                ensure_css_link(built_dir)

        elif built_dir.name == "main":
            version_html = version_template.render(
                versions=built_versions,
                current_version=None,
                latest=latest_tag,
            )
            banner_html = version_banner_template.render(
                current_version=None,
                latest=latest_tag,
            )
            recursive_patch_html_files(built_dir, version_html, "sidebar-tree")
            recursive_patch_html_files_add_banner(built_dir, banner_html)

            # Copy CSS and ensure link is present for all pages (correct depth)
            static_dir = built_dir / "_static" / "css"
            static_dir.mkdir(parents=True, exist_ok=True)
            with open(new_css, encoding="utf-8") as f:
                new_css_content = f.read()
            with open(
                static_dir / "version_selector.css", "w", encoding="utf-8"
            ) as f:
                f.write(new_css_content)
            ensure_css_link(built_dir)
