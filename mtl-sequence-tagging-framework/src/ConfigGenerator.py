"""
This module allows to generate configuration files from a template.
The configuration files are based on the intervals defined in the template and are randomly sampled from them.
"""
import os
import sys
import uuid
from distutils.dir_util import mkpath

import copy

import time
from ruamel import yaml

from shared_modules.random_search.TrialGenerator import INTERVAL_TYPE_LIST, Interval, TrialGenerator


class ConfigGenerator(object):
    """
    Objects of this class can be used to generate specific configuration files from templates.
    A template consists of a configuration file template with variables (strings) to replace.
    For each variable, a value interval needs to be specified.
    """
    def __init__(self, template_path, out_dir_path):
        """
        Initialize the generator.
        The generator loads the template from the specified path and ensures that the path `out_dir_path` exists by
        creating all missing folders along the path if necessary.

        Args:
            template_path (str): Path to the template
            out_dir_path (str): Path that indicates where generated configuration files shall be stored.
        """
        assert os.path.exists(template_path), "Expected the path %s to exist" % template_path
        assert isinstance(out_dir_path, str), "Expected `out_dir_path` to be a string"

        if not os.path.exists(out_dir_path):
            mkpath(out_dir_path)

        with open(template_path, mode="r") as f:
            variables_and_template = yaml.load_all(f, Loader=yaml.RoundTripLoader)

            # Convert generator to list
            variables_and_template = [x for x in variables_and_template]
        assert len(variables_and_template) == 2, "Expected the template file to contain two YAML documents"

        variables, template = variables_and_template

        assert isinstance(variables, dict), "Expected variables to be a dictionary"
        assert isinstance(template, dict), "Expected template to be a dictionary"

        self._trial_generator = TrialGenerator(self.variables_to_intervals(variables))
        self._template = template
        self._out_dir_path = out_dir_path
        self._template_path = template_path

    @staticmethod
    def variables_to_intervals(variables):
        """
        Helper method to convert interval specification dictionaries read from a YAML file into Interval objects.
        This conversion is necessary to use the TrialGenerator.

        Different interval types are distinguished by the "interval_type" property.

        Args:
            variables (dict): A dictionary of interval specifications. The key is always the variable name and the value
                is an interval specification, i.e. either a "list", "discrete" or "continuous" interval specification.

        Returns:
            `list` of Interval: A list of interval objects
        """
        assert isinstance(variables, dict)

        intervals = []

        for name, content in variables.items():
            assert isinstance(content, dict)
            assert "interval_type" in content, "Each variable needs to specify the interval type."

            if content["interval_type"] == INTERVAL_TYPE_LIST:
                intervals.append(Interval(
                    name,
                    interval_type=content["interval_type"],
                    values=content.get("values", None),
                ))
            else:
                intervals.append(Interval(
                    name,
                    interval_type=content["interval_type"],
                    start=content.get("start", None),
                    end=content.get("end", None),
                ))

        return intervals

    @staticmethod
    def update_template(template, trial):
        """
        Update the provided template object (may be a dictionary or a list) with values from the trial object.
        This method is recursive for dictionary and list values. If a string value is encountered, the method tries to
        replace the string with its corresponding value from the trial.

        NOTE: this method mutates the template object!

        Args:
            template (dict or list): Either a dictionary or a list of values.
            trial (dict): A mapping from variable names to values.

        Returns:
            dict: Updated template
        """
        assert isinstance(template, dict) or isinstance(template, list)
        items = template.items() if isinstance(template, dict) else enumerate(template)

        for key, value in items:
            if isinstance(value, str):
                if value in trial:
                    template[key] = trial[value]
            elif isinstance(value, dict) or isinstance(value, list):
                template[key] = ConfigGenerator.update_template(template[key], trial)

        return template

    def generate(self):
        """
        Generate a configuration file using the trial generator and store the file in the specified location.

        Returns:
            str: Path to the generated configuration file.
        """
        # Use a copy of the template to prevent modifying the original
        template = copy.deepcopy(self._template)
        trial = self._trial_generator.next()

        template = self.update_template(template, trial)
        file_name = "%s_%s_%s" % (
            time.strftime("%Y-%m-%d_%H%M"),
            uuid.uuid4().hex,
            os.path.basename(self._template_path)
        )
        file_path = os.path.join(self._out_dir_path, file_name)
        content = yaml.dump(template, Dumper=yaml.RoundTripDumper)

        with open(file_path, mode="w") as f:
            f.write(content)

        return file_path

if __name__ == "__main__":
    path_to_template = sys.argv[1]
    path_to_out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.abspath(".")

    gen = ConfigGenerator(path_to_template, path_to_out_dir)
    gen.generate()
