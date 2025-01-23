import time
import configparser
import os
import json
import numpy as np
import sys
import xarray as xr
from glob import glob
from importlib import import_module
from pymob.utils.misc import get_host
from pymob.utils.errors import errormsg

def create_fname_date_ver(dirname="./", fname="file", fextension=".txt"):
    ver = 0
    file = dirname + fname + time.strftime("%Y%m%d") + "_" + str(ver) + fextension
    while os.path.isfile(file):
        ver += 1
        file = dirname + fname + time.strftime("%Y%m%d") + "_" + str(ver) + fextension

    return file

def opt(a, b, c):
    """
    tests arguments for their state and returns with priority a, b and then c
    """
    return a if a is not None else b if b is not None else c

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

list_converter = lambda x: x if " " not in x else [i.strip() for i in x.split(' ')]

converters = {
    'list': list_converter,
    'strint': lambda x: int(x) if x.isdigit() else x,
    'strfloat': lambda x: float(x) if is_number(x) else x,
    'listfloat': lambda x: x if " " not in x else [float(i.strip()) for i in x.split(' ')]
}

def read_config(config_file):
    # get last bit of file name after dot
    ext = config_file.split(".")[-1]

    if ext == "json":
        with open(config_file, "r") as f:
            return json.load(f)

    if ext == "cfg":
        config = configparser.ConfigParser(
            converters=converters,
            interpolation=configparser.ExtendedInterpolation()
        )
        _ = config.read(config_file)
        return config

def parse_config_section(section, method: str="strint"):
    """
    method can be one of 'strint' (mix of string and integers) or "list"
    if lists are also in the section
    """
    getter = getattr(section, f"get{method}")
    return {k:getter(k) for k in section.keys()}

def store_sbi_simulations(path, theta, x, simname=None):
    if simname == None:
        simname = time.strftime("%Y%m%d%H%M")
    simloc = os.path.normpath(os.path.join(path, simname))
    os.makedirs(simloc, exist_ok=True)
    np.savetxt(os.path.join(simloc, "theta.txt"), theta.numpy())
    np.savetxt(os.path.join(simloc, "x.txt"), x.numpy())

def read_settings():
    try:
        return read_config("config/settings.json")
    except FileNotFoundError:
        return read_config("config/settings_default.json")

def _insert_username(settings, dir):
    username = settings["username_remote"]
    return dir.replace("_USERNAME_", username)

def prepare_sbi(sbi_config_file):
    config = read_config(sbi_config_file)
    sim_conf = read_config(config["simulation_config"])

    # get the correct file
    settings = read_settings()
    host = get_host(settings["localhosts"])
    output_dir = os.path.join(config[f"output_{host}"])
    
    output = _adapt_output_to_machine(output_dir=output_dir)

    os.makedirs(config["plots"], exist_ok=True)

    return config, sim_conf, output

def unixify_path(path):
    normed_path_os = os.path.normpath(path)
    unix_path = normed_path_os.replace(os.path.sep, "/")
    return unix_path

def scenario_file(file, case_study, scenario, pkg_dir="case_studies"):
    file_path = os.path.join(pkg_dir, case_study, "scenarios", scenario, file)
    if not os.path.exists(file_path):
        cwd = os.getcwd()
        raise FileNotFoundError(
            f"The scenario config file ({file}) was not found in this location: "
            f"'{file_path}'; your working directory is '{cwd}'. "
            "If your working directory is inside the location of your case_study, "
            "try navigating a few levels up the directory tree with "
            "`pkg_dir='..'` or `pkg_dir='../..'`. If your working directory is "
            "outside of the location of your case studies try `pkg_dir='case_studies'`. "
            "If nothing helps, simply provide the absolute path to the parent "
            "directory where your case study is located."
        )
    return file_path

def case_study_output(case_study, scenario, pkg_dir="case_studies"):
    study_name = os.path.basename(case_study)
    output = os.path.normpath(os.path.join(
        pkg_dir, study_name, "results", scenario))
    return pkg_dir, output

def prepare_scenario(case_study, scenario, input_files=[]):
    pkg_dir, output = case_study_output(case_study, scenario)
    
    paths_input_files = []
    for file in input_files:
        fp = scenario_file(file, case_study, scenario)
        paths_input_files.append(fp)

    output_path = reroute_output_to_base(output)

    return paths_input_files, output_path


def reroute_output_to_base(output):
    settings = read_settings()
    return unixify_path(os.path.join(
        settings[f"output_base"], 
        unixify_path(output)))

def prepare_casestudy(
    case_study: tuple[str,str], 
    config_file: str="settings.cfg", 
    pkg_dir="case_studies"
) -> configparser.ConfigParser:
    """Loads the configuration file of the case study by providing the name of 
    the case study directory, the name of the scenario directory, and the name
    of the configuration file. 
    Also adds the case_study directory to `sys.path` so that all modules in the
    root of the case_study directory can be imported (e.g. import sim, 
    import mod, ...)
    
    By default the function assumes that you have the following directory 
    structure:
    
    ```kotlin
    project_directory (working directory)
      ├─ case_studies
      │   ├─ case_study_1
      │   │   ├─ scenarios
      │   │   │   ├─ scenario_1
      │   │   │   │   └─ settings.cfg
      │   │   │   ├─ scenario_2
      │   │   │   │   └─ settings.cfg
      │   │   └─ results
      │   ├─ sim.py
      │   ├─ mod.py
      │   ├─ data.py
      │   ├─ plot.py
      │   └─ ...
      └─ other_directory
    ```
      
    Parameters
    ----------
    case_study : tuple[str,str]
        A tuple of the case_study direcory and the scenario directory
    config_file : str, optional
        The name of the configuration file contained in the scenario directory.
        Default is 'settings.cfg'
    pkg_dir : str, optional
        The name of the folder, where all case studies are located. Can also be
        a relative or an absolute file path. By default "case_studies"

    Returns
    -------
    ConfigParser
        The configuration file of the case-study-scenario combination.

    Example
    -------
    This example demonstrates how to use the prepare_casestudy function:

    ```python
    from my_module import prepare_casestudy

    # Example usage of prepare_casestudy function
    config = prepare_casestudy(("lotka_volterra_case_study", "test_scenario"), "settings.cfg")

    # Now you can access configuration settings
    print(config.get("case-study", "name"))
    print(config.get("case-study", "scenario"))

    # And use the config file to initialize a Simulation object
    from sim import Simulation
    sim = Simulation(config)
    ```
    """
    # read sbi-config, sim-config and settings
    assert len(case_study) == 2, "case_study must be a list with first entry <case_study_directory> and 2nd entry the scenario name"
    config = read_config(scenario_file(config_file, *case_study, pkg_dir=pkg_dir))
    
    config["case-study"]["root"] = os.getcwd()
    config["case-study"]["name"] = case_study[0]
    config["case-study"]["scenario"] = case_study[1]

    # store package dir
    # path_case_study = os.path.join(pkg_dir, case_study[0])
    config["case-study"]["package"] = pkg_dir

    # append relevant paths to sys
    root = os.path.join(config.get("case-study", "root"), pkg_dir)
    if root not in sys.path:
        sys.path.append(root)
    
    package = os.path.join(config.get("case-study", "root"), pkg_dir, case_study[0])
    if package not in sys.path:
        sys.path.append(package)

    return config

def prepare_casestudy_sbi(case_study):
    # read sbi-config, sim-config and settings
    config = read_config(scenario_file("sbi.json", *case_study))
    assert len(case_study) == 2, "case_study must be a list with first entry <case_study_directory> and 2nd entry the scenario name"
    sim_conf = prepare_scenario(*case_study)
    settings = read_settings()

    # determine directory depending of whether working local or remote
    host = get_host(settings["localhosts"])
    output_dir = unixify_path(os.path.join(
        settings[f"{host}_output_base"], 
        unixify_path(sim_conf["simulation"]["output"]).replace(
            "simulation", "sbi")
    ))

    # adapt output to the type of machine being worked on
    output = _adapt_output_to_machine(output_dir=output_dir)
    
    # create plot dir and pass to sbi-config
    plot_dir = unixify_path(sim_conf["simulation"]["output"]).replace(
        "simulation", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    config["plots"] = plot_dir
    
    return config, sim_conf, output

def _adapt_output_to_machine(output_dir):
    settings = read_settings()
    host = get_host(settings["localhosts"])

    if host == "remote":
        output_dir = os.path.normpath(_insert_username(settings, output_dir))

        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError:
            if os.path.isabs(output_dir):
                output_dir_old = output_dir
                output_dir = os.path.abspath(output_dir_old[1:])  # remove leading /
                sys.stdout.write(
                    f"no permission to write to {output_dir_old} using {output_dir}"
                )
                os.makedirs(output_dir, exist_ok=True)
    
    return unixify_path(output_dir)

def import_package(package_path):
    """
    this script handles the import of a case study without the typical 
    __init__.py file. It iterates through all .py files in the root directory
    of the case study (typically: sim, mod, stats, plot, data, prior)
    and imports them with ()
    package     path to package from project dir (e.g. "model/core/daphnia")
    """
    package = os.path.basename(package_path)
    package_dir = os.path.dirname(package_path)
    sys.path.append(package_dir)

    # it is for me only possible to import the specific modules and not 
    # only the package and then refer to the modules
    for mod in glob(os.path.join(package_dir, package, "*.py")):
        module = os.path.basename(mod).split('.')[0]
        try:
            _ = import_module(f"{package}.{module}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(errormsg(
                """The module could not be found. This can happen if the name
                of the case-study conflicts with another package. To avoid this
                conflict, give the case study a more unique name. E.g.
                case_study_test. You can confirm, whether another module with that name
                exists, by opening python in the command line and typing the
                name of your case study, if there is no error, it means that the
                package was found in your environment. You can then safely assume
                that the error came from a naming conflict and resolve it as suggested.
                """
            ))

    pkg = import_module(package)
    return pkg

def sequential_filename_iterator(path, sep="_"):
    """
    file storing function that appends a running number to the 
    filename, depending on the number of files in the directory. Note that this
    does only work if files are created sequentially. It will probably produce 
    errors if the files are produced in parallel.
    """
    # split path
    directory, file = os.path.split(path)

    # split filename
    filename, fileext = file.split(".")

    # find number of files in directory
    nfiles = len([
        f for f in os.listdir(directory) 
        if os.path.isfile(os.path.join(directory, f))
    ])
    filename = filename + sep + str(nfiles).zfill(5)

    newpath = os.path.join(directory, filename + "." + fileext)

    return newpath

def serialize(ds, convert_time=True):
    """
    function serializes a dataset. By default converts a time coordinate from
    datetime64[ns] to float[ns]
    """

    if convert_time:
        ds["time"] = ds.time.astype(float)

    return json.dumps(ds.to_dict())

def deserialize(dct, convert_time=True):
    ds = xr.Dataset.from_dict(json.loads(dct))

    if convert_time:
        ds["time"] = ds.time.astype("timedelta64[ns]")

    return ds

def unnest(d, flat, parent_key="", sep="."):
    for child_key, value in d.items():
        if parent_key != "":
            key = f"{parent_key}{sep}{child_key}"
        else:
            key = child_key
            
        if isinstance(value, dict):
            flat = unnest(d=value, flat=flat, parent_key=key)
        else:            
            flat.append((key, value))
    
    return flat


def go_to_case_studies(case_study_dir="case_studies", stop_inside_case_studies=True):
    """A convenience method to find the case studies directory from a directory
    within the case study. And change the working directory respectively"""
    cwd = os.getcwd()
    while not os.path.exists(os.path.join(os.getcwd(), case_study_dir)):
        if os.path.basename(os.getcwd()) == case_study_dir and stop_inside_case_studies:
            break

        os.chdir("..")  # navigate one level up

        if os.getcwd() == "/":
            os.chdir(cwd)
            raise NotADirectoryError(
                f"The case study directory {case_study_dir} could not be found"
                "by upwards traversing the directory tree. Please specify "
                "manually the location of the parent of your case_studies folder "
                "by using `os.chdir('path/to/the/parent/of/the/case/studies/folder')`"
            )
        
    print(f"Found case study directory in: {os.getcwd()}")
