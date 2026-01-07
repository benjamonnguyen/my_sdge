#!/usr/bin/env python3

import numpy as np
import pandas as pd
import datetime
import yaml
import traceback
import os
from functools import cache
from collections import namedtuple
import click
from plots import *

# for holiday exclusion
from pandas.tseries.holiday import USFederalHolidayCalendar


def load_yaml(filepath):
    """
    Load the yaml file. Returns an empty dictionary if the file cannot be read.
    """
    # yaml_path = os.path.join(pwd, filepath)
    try:
        with open(filepath, "r") as stream:
            dictionary = yaml.safe_load(stream)
            return dictionary
    except:
        traceback.print_exc()
        return dict()


def convert_12h_to_24h(time_str):
    dt = datetime.datetime.strptime(time_str, "%I:%M %p")
    # extract the hour
    time_24h_str = dt.strftime("%H")
    return int(time_24h_str)


def validate_dates(days):
    """
    To validate that the data is within one continuous year.
    """
    # days is sorted from low to high
    if days[0].date.year == days[-1].date.year:
        # all data from the same year
        pass
    if days[-1].date.year - days[0].date.year > 1:
        # this contains data from more than one year
        raise ValueError("Cannot use data from more than one year")
    if days[-1].date.year - days[0].date.year == 1:
        # span year n and year n+1
        if days[-1].date.month > days[0].date.month:
            # this contains data from more than one year
            # for example 2023-09 is more than 1 year from any day in 2022-08
            raise ValueError("Cannot use data from more than one year")
        elif days[-1].date.month == days[0].date.month:
            # starting from (y,m,d), you can get to (y+1,m,d-1) as the last day when d!=1
            if days[-1].date.day >= days[0].date.day:
                raise ValueError("Cannot use data from more than one year")


SDGEDay = namedtuple("SDGEDate", ["date", "season"])

pwd = os.path.dirname(os.path.realpath(__file__))


class SDGECaltulator:
    def __init__(
        self,
        daily_24h,
        rates,
        pcia_rates,
        zone="coastal",
        service_type="electric",
        pcia_year="2021",
        solar="NA",
    ):
        self.daily_24h = daily_24h
        self.days = [
            SDGEDay(date, get_season(date)) for date in extract_dates(self.daily_24h)
        ]
        self.zone = zone
        self.rates = rates
        self.pcia_rate = pcia_rates[int(pcia_year)]
        self.service_type = service_type
        self.total_usage = sum(
            [sum([x[1] for x in usage]) for date, usage in self.daily_24h.items()]
        )
        self.solar = solar

        # assert self.days[0].date.year == self.days[-1].date.year, "all data must be from the same year"
        validate_dates(self.days)
        self.print_info()

    def print_info(self):
        print(f"starting:{self.days[0].date} ending:{self.days[-1].date}")
        print(
            f"{len(self.days)} days, {len([x for x in self.days if x.season == 'summer'])} summer days, {len([x for x in self.days if x.season == 'winter'])} winter days"
        )
        if self.solar != "NA":
            print(f"solar setup: {self.solar}")
        print(f"total_usage:{self.total_usage:.4f} kWh")

    def generate_plots(self):
        # plot hourly data summed across days
        aggregated_hourly_net_usage_plot(daily=self.daily_24h)
        daily_net_usage_plot(daily=self.daily_24h)

    @cache
    def tally(self, schedule=None):
        daily_arrays = category_tally_by_schedule(
            daily=self.daily_24h, schedule=schedule
        )
        rates_classes = schedule.rates_classes

        season_days_counter = {"summer": 0, "winter": 0}
        # tally the summer usage and winter usage
        season_class_tally = {
            "summer": {x: 0.0 for x in rates_classes},
            "winter": {x: 0.0 for x in rates_classes},
        }
        for k, day in enumerate(self.days):
            season_days_counter[day.season] += 1
            for rate_class in rates_classes:
                season_class_tally[day.season][rate_class] += daily_arrays[rate_class][
                    k
                ]
        return rates_classes, season_days_counter, season_class_tally

    def calculate(self, plan=None):
        # usage tally
        plan_data = self.rates[plan]
        schedule = get_schedule_function(plan_data)
        rates_classes, season_days_counter, season_class_tally = self.tally(
            schedule=schedule
        )
        # print(season_class_tally)

        total_fee = 0.0

        # Iterate through available seasons dynamically
        for season, season_data in plan_data.items():
            # Skip non-season fields (like tou_type)
            if season not in season_class_tally:
                continue

            season_total_usage = sum(season_class_tally[season].values())

            # Calculate rates by summing tariffs and eecc for each rate class
            rates_by_class = {}
            for rate_class in rates_classes:
                if rate_class in season_data:
                    rate_data = season_data[rate_class]
                    rates_by_class[rate_class] = rate_data["tariffs"] + rate_data["eecc"]

            total_fee += get_raw_sum(season_class_tally[season], rates_by_class)

            # Handle baseline adjustment credit (if present at season level)
            credit_per_kwh = 0.0
            if "baseline_adjustment_credit" in season_data:
                # Note: baseline_adjustment_credit is negative in new schema
                credit_per_kwh = -season_data["baseline_adjustment_credit"]

            allowance_deduction = get_allowance_deduction(
                zone=self.zone,
                season=season,
                service_type=self.service_type,
                billing_days=season_days_counter[season],
                total_usage=season_total_usage,
                credit_per_kwh=credit_per_kwh,
            )
            # remove the deduction
            total_fee -= allowance_deduction

        # Note: service_fee not present in new schema, assuming 0 for now
        # apply the PCIA rates for CCA
        if "CCA" in plan:
            total_fee += self.total_usage * self.pcia_rate
        return total_fee


def calculate_misc_fees(total_usage=0.0, pcia_rate=0.01687):
    misc_fee = 0.0

    return misc_fee


def get_raw_sum(usage_by_class, rates_by_class):
    """
    usage_by_class (dict)
    rates_by_class (dict)
    """
    return sum(
        [
            usage_by_class[rates_class] * rates_by_class[rates_class]
            for rates_class in usage_by_class
        ]
    )


@cache
def get_allowance_deduction(
    zone="coastal",
    season=None,
    service_type="electric",
    billing_days=30,
    total_usage=0.0,
    credit_per_kwh=0.11724,
):
    # calculate 130% allowance deduction
    baseline130 = get_baseline(
        zone=zone,
        season=season,
        service_type=service_type,
        multiplier=1.3,
        billing_days=billing_days,
    )
    # for non-solar users, and solar users with net consumption (more consumption than generation)
    if total_usage > 0:
        deducted_usage = min(total_usage, baseline130)
    # for solar users with net generation (more generation than consumption), the credit would be negative
    else:
        deducted_usage = max(total_usage, -baseline130)
    # calculate deduction
    allowance_deduction = credit_per_kwh * deducted_usage
    return allowance_deduction


@cache
def get_baseline(
    zone=None, season=None, service_type="electric", multiplier=1.3, billing_days=30
):
    # source: https://www.sdge.com/baseline-allowance-calculator
    zone_index_mapping = {"coastal": 0, "inland": 1, "mountain": 2, "desert": 3}
    zone_index = zone_index_mapping[zone]

    summer_electric = [6, 8.7, 15.2, 17]
    winter_electric = [8.8, 12.2, 22.1, 17.1]

    summer_combined = [9.0, 10.4, 13.6, 15.9]
    winter_combined = [9.2, 9.6, 12.9, 10.9]

    daily_baseline = {
        "electric": {
            "summer": summer_electric,
            "winter": winter_electric,
        },
        "combined": {
            "summer": summer_combined,
            "winter": winter_combined,
        },
    }
    return int(
        np.floor(
            multiplier * billing_days * daily_baseline[service_type][season][zone_index]
        )
    )


def extract_dates(daily_24h):
    """
    Extract dates from the daily_24h dictionary
    """
    return [pd.to_datetime(date, format="%Y-%m-%d").date() for date in daily_24h.keys()]


def get_season(date):
    if date.month in {6, 7, 8, 9, 10}:
        return "summer"
    return "winter"


# https://www.sdge.com/regulatory-filing/16026/residential-time-use-periods
@cache
def schedule_sop(date):
    """
    rates schedule for plans with SUPER OFFPEAK, OFFPEAK, PEAK rates
    """
    is_march_or_april = 1 if (date.month == 3 or date.month == 4) else 0

    # non-holiday weekdays
    WEEKDAY_HOURS = {
        "super_offpeak": {0, 1, 2, 3, 4, 5},
        "offpeak": {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23},
        "onpeak": {16, 17, 18, 19, 20},
    }
    # weekends and holidays
    HOLIDAY_HOURS = {
        "super_offpeak": {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
        "offpeak": {14, 15, 21, 22, 23},
        "onpeak": {16, 17, 18, 19, 20},
    }

    if is_march_or_april:
        WEEKDAY_HOURS["super_offpeak"] = {0, 1, 2, 3, 4, 5, 10, 11, 12, 13}
        WEEKDAY_HOURS["offpeak"] = {6, 7, 8, 9, 14, 15, 21, 22, 23}

    # which day is it?
    weekday = date.weekday()

    # mark US holidays
    holidays = holidays_of_year(date.year)

    if weekday == 5 or weekday == 6 or date in holidays:
        return HOLIDAY_HOURS
    return WEEKDAY_HOURS


@cache
def holidays_of_year(year):
    cal = USFederalHolidayCalendar()
    start = datetime.datetime(year, 1, 1)
    end = datetime.datetime(year + 1, 1, 1)
    holidays = cal.holidays(start=start, end=end).to_pydatetime()
    return holidays


@cache
def schedule_op(date):
    """
    rates schedule for plans with OFFPEAK, PEAK rates
    """
    EVERYDAY_HOURS = {
        "offpeak": {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23},
        "onpeak": {16, 17, 18, 19, 20},
    }
    return EVERYDAY_HOURS


@cache
def schedule_flat(date):
    """
    rates schedule for flat rate plans (non-TOU)
    """
    EVERYDAY_HOURS = {"flat": {i for i in range(24)}}
    return EVERYDAY_HOURS


schedule_sop.rates_classes = ["super_offpeak", "offpeak", "onpeak"]
schedule_op.rates_classes = ["offpeak", "onpeak"]
schedule_flat.rates_classes = ["flat"]


def get_schedule_function(plan_data):
    """
    Map plan type to schedule function based on new schema.
    If plan has tou_type field, it's a TOU schedule, otherwise it's flat.
    """
    if "tou_type" in plan_data:
        tou_type = plan_data["tou_type"]
        type_to_schedule = {
            "sop": schedule_sop,
            "op": schedule_op,
        }
        return type_to_schedule.get(tou_type)
    else:
        # No tou_type means it's a flat rate schedule
        return schedule_flat


def category_tally_by_schedule(daily=None, schedule=None):
    """
    Returns the daily sum of usage for each tou category in a dictionary.
    """
    daily_arrays = {l: np.array([]) for l in schedule.rates_classes}

    for date, consumption_data in daily.items():
        d = pd.to_datetime(date, "%Y-%m-%d").date()

        for category in daily_arrays:
            current_array = daily_arrays[category]
            # remove assumption about number of data items
            daily_arrays[category] = np.append(
                current_array,
                sum(
                    [
                        consumption_data[i][1]
                        for i in range(len(consumption_data))
                        if consumption_data[i][0] in schedule(d)[category]
                    ]
                ),
            )

    return daily_arrays


def load_df(filename):
    # read the csv and skip the first rows
    df = pd.read_csv(
        filename,
        skiprows=13,
        index_col=False,
        usecols=["Date", "Start Time", "Duration", "Consumption", "Net"],
        skipinitialspace=True,
        dtype={"Consumption": np.float32},
        parse_dates=["Date"],
    )
    return df


@click.command()
@click.option(
    "-f",
    "--filename",
    required=True,
    help="The full path of the 60-minute exported electricity usage file.",
)
@click.option(
    "-z",
    "--zone",
    default="coastal",
    type=click.Choice(["coastal", "inland", "mountain", "desert"]),
    show_default=True,
    help="The climate zone of the house.",
)
@click.option(
    "-s",
    "--solar",
    default="NA",
    type=click.Choice(["NA", "NEM1.0"]),
    show_default=True,
    help="The solar setup.",
)
@click.option(
    "--pcia_year",
    default="2021",
    type=click.Choice([str(x) for x in range(2009, 2024)]),
    show_default=True,
    help="The vintage of the PCIA fee. (indicated on the bill)",
)
def plot_sdge_hourly(filename, zone, pcia_year, solar):
    df = load_df(filename)

    interval = df.iloc[0]["Duration"]
    # convert the 12h-format start time to 24h-format
    df["Start Time"] = pd.to_datetime(df["Start Time"], format="%I:%M %p").dt.strftime(
        "%H"
    )
    # convert hour to int index
    df["Start Time"] = df["Start Time"].astype(int)

    if solar == "NA":
        consumption_column_label = "Consumption"
    elif solar == "NEM1.0":
        consumption_column_label = "Net"

    # occasionally there are two readings for the same time slot, for now, we sum up the duplicates #TODO: ask SDGE what's happening!
    # df = df.drop_duplicates(subset=["Date","Start Time"], keep="last")
    # this step sums duplicates for 60-min interval data; aggregates the 15-min interval data into hourly data
    df = (
        df.astype("object")
        .groupby(["Date", "Start Time"], as_index=False, sort=False)
        .agg("sum")
    )  # use astype to prevent pd from converting int to float
    daily = df.groupby("Date")[["Start Time", consumption_column_label]].apply(
        lambda x: tuple(x.values)
    )  # sorted by date by default

    # tou_stacked_plot(daily=daily, plan="TOU-DR1")

    # plot day by day
    # daily_hourly_2d_plot(daily=daily)
    # daily_hourly_3d_plot(daily=daily)

    plans_and_charges = dict()
    sdge_schedules = os.path.join(pwd, "rates", "sdge_schedules.yaml")
    pcia_file = os.path.join(pwd, "rates", "pcia.yaml")

    rates = load_yaml(sdge_schedules)
    pcia_rates = load_yaml(pcia_file)

    # Create list of plans (excluding CCA plans for now)
    plans = [plan for plan in rates.keys() if not plan.startswith("CCA")]

    # @implement simplify by directly passing in pcia_rates[pcia_year]
    c = SDGECaltulator(
        daily, rates, pcia_rates, zone=zone, pcia_year=pcia_year, solar=solar
    )

    # iterate through loaded schedules
    for plan in plans:
        estimated_charge = c.calculate(plan=plan)
        plans_and_charges[plan] = estimated_charge

    for item in sorted(plans_and_charges.items(), key=lambda x: x[1]):
        print(f"{item[0]:<15} ${item[1]:.4f} ${item[1] / c.total_usage:.4f}/kWh")

    c.generate_plots()


if __name__ == "__main__":
    # print(get_baseline(zone="coastal", season="summer", service_type="electric", multiplier=1.3, billing_days=29))

    plot_sdge_hourly()
