"""Convert a folder of data to a common format."""

import csv
import logging
import re
import pathlib
import time
import uuid

from collections import defaultdict

from gooey import Gooey, GooeyParser

# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import scipy.interpolate
import yaml

# number of points to average over at the end of a stabilisation scan to record as the
# stabilised value
POINTS_TO_AVERAGE = 10

SETUP_PREFIX = r"IV_pixel_setup_"
SETUP_EXT = r"csv"
SETUP_PATTERN = re.compile(SETUP_PREFIX + r"(\d+)\." + SETUP_EXT)

RUN_ARGS_PREFIX = r"run_args"
RUN_ARGS_EXT = r"yaml"
RUN_ARGS_PATTERN = re.compile(RUN_ARGS_PREFIX + r"(\d+)\." + RUN_ARGS_EXT)

PROCESSED_FOLDER_NAME = "processed"


@Gooey(
    dump_build_config=False,
    program_name="Summary Generator",
    default_size=(750, 530),
    header_bg_color="#7B7B7B",
)
def parse():
    """Parse command line arguments to Gooey GUI."""
    desc = "Create a summary file from solar cell data"

    parser = GooeyParser(description=desc)
    req = parser.add_argument_group(gooey_options={"columns": 1})
    req.add_argument(
        "folder",
        metavar="Folder containing data to be processed",
        help="Absolute path to the folder containing measurement data",
        widget="DirChooser",
    )
    req.add_argument(
        "--stack_ivs",
        metavar="Stack IVs",
        help="Stack IVs into single files grouped by common slot, label, device, and illumination condition",
        widget="CheckBox",
        action="store_true",
    )
    req.add_argument(
        "--debug",
        metavar="DEBUG",
        help="Export debug info to a file",
        widget="CheckBox",
        action="store_true",
    )
    return parser.parse_args()


def create_logger(
    level: int,
    log_folder: pathlib.Path,
    name: str | None = None,
):
    logging.captureWarnings(True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create a filter to remove messages from certain imports
    class ImportFilter(logging.Filter):
        """Filter log records from named third-party imports."""

        def filter(self, record: logging.LogRecord) -> bool:
            if record.name.startswith("matplotlib"):
                return False
            elif record.name.startswith("PIL"):
                return False
            else:
                return True

    # console logger
    ch = logging.StreamHandler()
    ch.addFilter(ImportFilter())
    logger.addHandler(ch)

    if level == logging.DEBUG:
        log_format = logging.Formatter(
            "%(asctime)s|%(name)s|%(levelname)s|%(filename)s:%(lineno)d|%(funcName)s|%(message)s"
        )
        # file logger
        fh = logging.FileHandler(log_folder.joinpath(f"{int(time.time())}.log"))
        fh.setFormatter(log_format)
        fh.addFilter(ImportFilter())
        logger.addHandler(fh)

    return logger


def get_setup_dict(data_folder: pathlib.Path) -> dict:
    """Generate dictionary of setup .csv files.

    Parameters
    ----------
    data_folder : pathlib.Path
        Folder containing measurement data.

    Returns
    -------
    setup_dict : dict
        Dictionary of setup info.
    """
    setup_dict = {}
    for file in data_folder.glob(f"{SETUP_PREFIX}*.{SETUP_EXT}"):
        match = SETUP_PATTERN.search(file.name)
        if match:
            timestamp = match.group(1)
            pixel_setup = pd.read_csv(
                data_folder.joinpath(f"{SETUP_PREFIX}{timestamp}.{SETUP_EXT}")
            )
            setup_dict[timestamp] = pixel_setup

    return setup_dict


def load_run_args(path: pathlib.Path):
    """Load run arguments from a yaml file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the run_args yaml file.

    Returns
    -------
    run_args : dict
        Run arguments dictionary.
    """

    class CustomLoader(yaml.SafeLoader):
        """Subclass safe loader to avoid modifying it inplace."""

    def construct_uuid(loader, node):
        mapping = loader.construct_mapping(node)
        return uuid.UUID(int=mapping["int"])

    CustomLoader.add_constructor(
        "tag:yaml.org,2002:python/object:uuid.UUID", construct_uuid
    )

    with open(path, encoding="utf-8") as open_file:
        run_args = yaml.load(open_file, Loader=CustomLoader)

    return run_args


def get_run_args_dict(data_folder: pathlib.Path) -> dict:
    """Generate dictionary of setup .csv files.

    Parameters
    ----------
    data_folder : pathlib.Path
        Folder containing measurement data.

    Returns
    -------
    run_args_dict : dict
        Dictionary of run arguments info.
    """
    run_args_dict = {}
    for file in data_folder.glob(f"{RUN_ARGS_PREFIX}*.{RUN_ARGS_EXT}"):
        match = RUN_ARGS_PATTERN.search(file.name)
        if match:
            timestamp = match.group(1)
            run_args = load_run_args(
                data_folder.joinpath(f"{RUN_ARGS_PREFIX}{timestamp}.{RUN_ARGS_EXT}")
            )
            run_args_dict[timestamp] = run_args

    return run_args_dict


def dummy_interpolation(anything):
    """Replace interpolation with nan when interpolation fails."""
    return np.nan


def generate_summary(data_folder: pathlib.Path, logger: logging.Logger):
    """
    Process data and generate a summary file.

    Parameters
    ----------
    data_folder : pathlib.Path
        Folder containing measurement data.
    logger : logging.Logger
        Logger object.
    """
    logger.info("Processing files and generating summary...")

    processed_folder = data_folder.joinpath(PROCESSED_FOLDER_NAME)
    processed_folder.mkdir(exist_ok=True)

    processed_header = [
        "voltage (V)",
        "current (A)",
        "time (s)",
        "status",
        "power (mW)",
        "current_density (mA/cm^2)",
        "power_density (mW/cm^2)",
    ]

    summary_header = [
        "slot",
        "label",
        "device",
        "area",
        "i_sc (mA)",
        "v_oc (V)",
        "p_max (mW)",
        "i_mp (mA)",
        "v_mp (V)",
        "ff",
        "j_sc (mA/cm^2)",
        "pd_max (mW/cm^2)",
        "j_mp (mA/cm^2)",
        "r_s (ohms)",
        "r_sh (ohms)",
        "i_ss (mA)",
        "v_ss (V)",
        "p_ss (mW)",
        "j_ss (mA/cm^2)",
        "pd_ss (mW/cm^2)",
        "scan_#",
        "kind",
        "filename",
    ]

    # create summary file
    summary_file = processed_folder.joinpath("summary.tsv")
    with open(summary_file, "w", newline="\n", encoding="utf-8") as open_file:
        writer = csv.writer(open_file, delimiter="\t")
        writer.writerow(summary_header)

    # get metadata dictionaries
    run_args_dict = get_run_args_dict(data_folder)
    setup_dict = get_setup_dict(data_folder)

    for ix, file in enumerate(data_folder.glob("*.tsv")):
        logger.info(f"Processing file {ix}: {file.name}")

        try:
            slot, label, device, experiment_timestamp_ext = file.name.split("_")
        except ValueError:
            # the device label probably wasn't provided
            slot, device, experiment_timestamp_ext = file.name.split("_")
            label = "-"

        experiment_timestamp, ext1, ext2 = experiment_timestamp_ext.split(".")
        meas_kind = re.sub(r"\d+", "", ext1)
        _pixel = int(device.replace("device", ""))
        _area_type = "dark_area" if "div" in ext1 else "area"

        try:
            _pixel_setup = setup_dict[experiment_timestamp][
                setup_dict[experiment_timestamp]["pad"] == _pixel
            ]
            _area = _pixel_setup[_pixel_setup["slot"] == slot].iloc[0][_area_type]
        except KeyError:
            # probably old style pixel setup file
            _pixel_setup = setup_dict[experiment_timestamp][
                setup_dict[experiment_timestamp]["mux_index"] == _pixel
            ]
            _area = _pixel_setup[_pixel_setup["system_label"] == slot].iloc[0][
                _area_type
            ]

        # load and process raw data
        data = np.genfromtxt(file, delimiter="\t", skip_header=1)
        if data.ndim == 1:
            # data only has one row so need to reshape as 2D array
            data = np.expand_dims(data, axis=0)
        _voltage = data[:, 0]
        _current = data[:, 1]
        _time = data[:, 2]
        _status = data[:, 3]
        _power = _current * _voltage * 1000
        _current_density = _current * 1000 / _area
        _power_density = _power / _area

        processed_data = np.column_stack(
            (
                _voltage,
                _current,
                _time,
                _status,
                _power,
                _current_density,
                _power_density,
            )
        )

        processed_data_list = list(processed_data)

        # write processed data file
        processed_file = processed_folder.joinpath(f"processed_{file.name}")
        with open(processed_file, "w", newline="\n", encoding="utf-8") as open_file:
            writer = csv.writer(open_file, delimiter="\t")
            writer.writerow(processed_header)
            writer.writerows(processed_data_list)

        # apply special formatting to suns_voc voc file if applicable
        _rel_time = processed_data[:, 2] - processed_data[0, 2]
        try:
            if ("vt" in ext1) and (
                run_args_dict[experiment_timestamp]["suns_voc"] >= 3
            ):
                # take first portion of voc dwell as ss-voc measurement
                mask = np.where(
                    _rel_time <= run_args_dict[experiment_timestamp]["i_dwell"]
                )
            elif ("vt" in ext1) and (
                run_args_dict[experiment_timestamp]["suns_voc"] <= -3
            ):
                # take last portion of voc dwell as ss-voc measurement
                mask = np.where(
                    _rel_time
                    >= _rel_time[-1] - run_args_dict[experiment_timestamp]["i_dwell"]
                )
            else:
                mask = [True] * len(processed_data[:, 0])
        except KeyError:
            # suns_voc key probably isn't available for this version of run_args
            mask = [True] * len(processed_data[:, 0])

        rel_time = _rel_time[mask]
        meas_voltage = processed_data[:, 0][mask]
        meas_current = processed_data[:, 1][mask]
        time_data = processed_data[:, 2][mask]
        status = processed_data[:, 3][mask]
        meas_p = processed_data[:, 4][mask]
        meas_j = processed_data[:, 5][mask]
        meas_pd = processed_data[:, 6][mask]

        # measurements not in compliance
        try:
            n_compliance = [not (int(format(int(s), "024b")[-4])) for s in status]
        except IndexError:
            n_compliance = [True for _ in status]
            logger.warning(
                "WARNING: Invalid status byte format so can't determine "
                + "measurements in compliance."
            )

        liv = "liv" in ext1
        div = "div" in ext1

        if "vt" in ext1:
            isc = np.nan
            voc = np.nan
            pmax = np.nan
            imp = np.nan
            vmp = np.nan
            ff = np.nan
            jsc = np.nan
            pdmax = np.nan
            jmp = np.nan
            rs = np.nan
            rsh = np.nan
            iss = np.nan
            vss = np.mean(meas_voltage[-POINTS_TO_AVERAGE:])
            pss = np.nan
            jss = np.nan
            pdss = np.nan
            scan_n = 1
        elif "mpp" in ext1:
            isc = np.nan
            voc = np.nan
            pmax = np.nan
            imp = np.nan
            vmp = np.nan
            ff = np.nan
            jsc = np.nan
            pdmax = np.nan
            jmp = np.nan
            rs = np.nan
            rsh = np.nan
            iss = np.mean(meas_current[-POINTS_TO_AVERAGE:]) * 1000
            vss = np.mean(meas_voltage[-POINTS_TO_AVERAGE:])
            pss = np.mean(meas_p[-POINTS_TO_AVERAGE:])
            jss = np.mean(meas_j[-POINTS_TO_AVERAGE:])
            pdss = np.mean(meas_pd[-POINTS_TO_AVERAGE:])
            scan_n = 1
        elif "it" in ext1:
            isc = np.nan
            voc = np.nan
            pmax = np.nan
            imp = np.nan
            vmp = np.nan
            ff = np.nan
            jsc = np.nan
            pdmax = np.nan
            jmp = np.nan
            rs = np.nan
            rsh = np.nan
            iss = np.mean(meas_current[-POINTS_TO_AVERAGE:]) * 1000
            vss = np.nan
            pss = np.nan
            jss = np.mean(meas_j[-POINTS_TO_AVERAGE:])
            pdss = np.nan
            scan_n = 1
        elif div or liv:
            try:
                f_i = scipy.interpolate.interp1d(
                    meas_voltage[n_compliance],
                    meas_current[n_compliance],
                    kind="linear",
                    bounds_error=False,
                    fill_value=0,
                )
            except ValueError:
                f_i = dummy_interpolation

            try:
                f_v = scipy.interpolate.interp1d(
                    meas_j[n_compliance],
                    meas_voltage[n_compliance],
                    kind="linear",
                    bounds_error=False,
                    fill_value=0,
                )
            except ValueError:
                f_v = dummy_interpolation

            dpdv = np.gradient(meas_p, meas_voltage)
            try:
                f_dpdv = scipy.interpolate.interp1d(
                    dpdv[n_compliance],
                    meas_voltage[n_compliance],
                    kind="linear",
                    bounds_error=False,
                    fill_value=0,
                )
            except ValueError:
                f_dpdv = dummy_interpolation

            r_diff = np.gradient(meas_voltage, meas_current)
            try:
                f_r_diff = scipy.interpolate.interp1d(
                    meas_voltage[n_compliance],
                    r_diff[n_compliance],
                    kind="linear",
                    bounds_error=False,
                    fill_value=0,
                )
            except ValueError:
                f_r_diff = dummy_interpolation

            if liv:
                isc = f_i(0) * 1000
                voc = f_v(0)
                vmp = f_dpdv(0)
                imp = f_i(vmp) * 1000

                try:
                    rs = f_r_diff(voc)
                    rsh = f_r_diff(0)
                except NameError:
                    rs = np.nan
                    rsh = np.nan

                scan_n = int(ext1.strip("liv"))
            else:
                isc = np.nan
                voc = np.nan
                vmp = np.nan
                imp = np.nan
                rs = np.nan
                rsh = np.nan

                scan_n = int(ext1.strip("div"))

            if isc != np.nan and vmp != np.nan and imp != np.nan:
                pmax = -imp * vmp
                ff = pmax / (isc * voc)
                jsc = isc / _area
                pdmax = pmax / _area
                jmp = imp / _area
            else:
                pmax = np.nan
                ff = np.nan
                jsc = np.nan
                pdmax = np.nan
                jmp = np.nan

            iss = np.nan
            vss = np.nan
            pss = np.nan
            jss = np.nan
            pdss = np.nan
        else:
            logger.warning(f"Invalid file extension: {ext1}.")

            isc = np.nan
            voc = np.nan
            pmax = np.nan
            imp = np.nan
            vmp = np.nan
            ff = np.nan
            jsc = np.nan
            pdmax = np.nan
            jmp = np.nan
            rs = np.nan
            rsh = np.nan
            iss = np.nan
            vss = np.nan
            pss = np.nan
            jss = np.nan
            pdss = np.nan
            scan_n = np.nan

        summary_data = [
            slot,
            label,
            _pixel,
            _area,
            isc,
            voc,
            pmax,
            imp,
            vmp,
            ff,
            jsc,
            pdmax,
            jmp,
            rs,
            rsh,
            iss,
            vss,
            pss,
            jss,
            pdss,
            scan_n,
            meas_kind,
            processed_file.name,
        ]

        # write summary data to file
        with open(summary_file, "a", newline="\n", encoding="utf-8") as open_file:
            writer = csv.writer(open_file, delimiter="\t")
            writer.writerow(summary_data)

    logger.info(
        f"File processing complete! Processed files can be found in: {str(processed_folder)}"
    )
    logger.info(
        f"Summary generation complete! Summary file can be found at: {str(summary_file)}"
    )


def stack_ivs(data_folder: pathlib.Path, logger: logging.Logger):
    """
    Stack I-V curves column-wise given a folder containing I-V data.

    Look at all files and stack those where slot, label, device#, and condition are
    common. Timestamp may differ, allowing cases where a run was stopped, then
    restarted, without moving a device.

    Parameters
    ----------
    data_folder : pathlib.Path
        Folder containing measurement data.
    logger : logging.Logger
        Logger object.
    """
    logger.info("Stacking IV's...")

    processed_folder = data_folder.joinpath(PROCESSED_FOLDER_NAME)

    # Regex pattern to extract slot, label, device#, timestamp, condition, and scan#
    # Parentheses determine what's captured
    pattern = re.compile(
        r"^processed_([^_]+)_([^_]+)_device(\d+)_(\d+)\.(liv|div)(\d+)\.tsv$"
    )

    # Dictionary to store files grouped by (slot, label, device#, condition)
    grouped_files = defaultdict(list)

    # Collect and group filenames
    for file in processed_folder.glob("processed_*_*_device*_*.*.tsv"):
        match = pattern.match(file.name)
        if match:
            slot, label, device, timestamp, condition, scan = match.groups()
            grouped_files[(slot, label, device, condition)].append((file.name))

    # Process each group separately
    for (slot, label, device, condition), filenames in grouped_files.items():
        logger.info(f"Grouping: {slot}, {label}, {device}, {condition}")

        # Sort files by timestamp and scan number
        filenames.sort()

        # Define the output file for this group
        output_file = processed_folder.joinpath(
            f"processed_{slot}_{label}_device{device}.{condition}-stacked.tsv"
        )

        # Read files and stack column-wise
        dataframes = []
        for filename in filenames:
            logger.info(f"Filename: {filename}")
            df = pd.read_csv(
                processed_folder.joinpath(filename), sep="\t", header=None, dtype=str
            )

            # Insert the filename as the first row
            filename_row = pd.DataFrame(
                [[filename] + [""] * (df.shape[1] - 1)], columns=df.columns
            )

            # Concatenate filename, header, and data row-wise
            full_df = pd.concat([filename_row, df], axis=0, ignore_index=True)

            # Append the full dataframe to the list
            dataframes.append(full_df)

        # Concatenate and save output
        stacked_df = pd.concat(dataframes, axis=1)
        stacked_df.to_csv(output_file, sep="\t", index=False)
        logger.info(f"New stacked file: {output_file.name}")


if __name__ == "__main__":
    args = parse()

    if args.debug:
        LOG_LEVEL = logging.DEBUG
    else:
        LOG_LEVEL = logging.INFO

    processed_folder = pathlib.Path(args.folder).joinpath(PROCESSED_FOLDER_NAME)
    processed_folder.mkdir(exist_ok=True)
    logger = create_logger(LOG_LEVEL, processed_folder)

    generate_summary(pathlib.Path(args.folder), logger)

    if args.stack_ivs:
        stack_ivs(pathlib.Path(args.folder), logger)
