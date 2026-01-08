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


def generate_cca_plans(sdge_rates, cca_rates):
    """
    Generate CCA-* plans from SDGE plans using CCA rates.

    For CCA plans:
    - Swap eecc with cca_eecc from CCA rates
    - Keep all other tariffs the same
    - Plans prefixed with CCA_ will automatically get PCIA fee added

    Logs if no corresponding SDGE plan is found for a CCA schedule.

    Returns dictionary with CCA-* plans.
    """
    import copy

    cca_plans = {}

    for cca_plan_name, cca_data in cca_rates.items():
        if cca_plan_name in sdge_rates:
            # Create CCA plan by deep copying SDGE plan
            cca_plan_data = copy.deepcopy(sdge_rates[cca_plan_name])

            # Replace eecc with cca_eecc for matching rate classes
            for season in ["summer", "winter"]:
                if season in cca_plan_data and season in cca_data:
                    sdge_season_data = cca_plan_data[season]
                    cca_season_data = cca_data[season]

                    # Swap eecc with cca_eecc for each matching rate class
                    for rate_class, cca_rate in cca_season_data.items():
                        if rate_class in sdge_season_data and isinstance(
                            sdge_season_data[rate_class], dict
                        ):
                            sdge_season_data[rate_class]["eecc"] = cca_rate

            cca_plans[f"CCA-{cca_plan_name}"] = cca_plan_data
        else:
            print(
                f"Warning: No corresponding SDGE plan found for CCA schedule '{cca_plan_name}'"
            )

    return cca_plans


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
        pcia_rate,
        zone="coastal",
        service_type="electric",
        solar="NA",
    ):
        self.daily_24h = daily_24h
        self.days = [
            SDGEDay(date, get_season(date)) for date in extract_dates(self.daily_24h)
        ]
        self.zone = zone
        self.rates = rates
        self.pcia_rate = pcia_rate
        self.service_type = service_type
        self.total_usage = sum(
            [sum([x[1] for x in usage]) for date, usage in self.daily_24h.items()]
        )
        self.solar = solar

        # assert self.days[0].date.year == self.days[-1].date.year, "all data must be from the same year"
        validate_dates(self.days)

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

    @cache
    def detailed_tally(self, schedule=None):
        """
        Calculate detailed breakdown by season, day type (weekday vs weekend/holiday), and TOU period.
        Returns: dict with structure:
        {
            "summer": {
                "weekday": {"days": N, "usage": {rate_class: kwh, ...}},
                "weekend": {"days": N, "usage": {rate_class: kwh, ...}}
            },
            "winter": {...}
        }
        """
        rates_classes = schedule.rates_classes

        # Initialize structure
        breakdown = {}
        for season in ["summer", "winter"]:
            breakdown[season] = {
                "weekday": {"days": 0, "usage": {x: 0.0 for x in rates_classes}},
                "weekend": {"days": 0, "usage": {x: 0.0 for x in rates_classes}},
            }

        # Process each day
        for k, day in enumerate(self.days):
            date = day.date
            season = day.season

            # Determine if weekday or weekend/holiday
            weekday = date.weekday()
            holidays = holidays_of_year(date.year)
            is_weekend = weekday == 5 or weekday == 6 or date in holidays
            day_type = "weekend" if is_weekend else "weekday"

            # Get day's schedule
            day_schedule = schedule(date)

            # Count days
            breakdown[season][day_type]["days"] += 1

            # Get consumption data for this day
            date_str = date.strftime("%Y-%m-%d")
            if date_str in self.daily_24h:
                consumption_data = self.daily_24h[date_str]

                # Tally usage by rate class
                for rate_class in rates_classes:
                    usage = sum(
                        [
                            consumption_data[i][1]
                            for i in range(len(consumption_data))
                            if consumption_data[i][0] in day_schedule[rate_class]
                        ]
                    )
                    breakdown[season][day_type]["usage"][rate_class] += usage

        return breakdown

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
                    rates_by_class[rate_class] = (
                        rate_data["tariffs"] + rate_data["eecc"]
                    )

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

        if "CCA" in plan:
            total_fee += self.total_usage * self.pcia_rate
        return total_fee


def print_detailed_analysis(
    plan,
    plan_data,
    breakdown,
    total_usage,
    total_cost,
    zone="coastal",
    service_type="electric",
):
    """
    Print detailed breakdown in hierarchical format.
    """
    print("=" * 80)
    print(plan)
    print("=" * 80)
    print()

    # Get schedule to determine rate classes
    schedule = get_schedule_function(plan_data)
    rates_classes = schedule.rates_classes

    # Calculate plan subtotal and baseline credits
    plan_subtotal = 0.0
    total_baseline_credit = 0.0

    # Process each season that has data
    for season, season_data in plan_data.items():
        # Skip non-season fields
        if season not in breakdown:
            continue

        # Calculate season total kWh
        season_kwh = sum(
            [
                breakdown[season][day_type]["usage"][rate_class]
                for day_type in ["weekday", "weekend"]
                for rate_class in rates_classes
            ]
        )

        # Count total days in season
        total_days = (
            breakdown[season]["weekday"]["days"] + breakdown[season]["weekend"]["days"]
        )

        # Pre-calculate season total cost for $/day display
        season_total_cost_precalc = 0.0
        for day_type in ["weekday", "weekend"]:
            for rate_class in rates_classes:
                if rate_class in season_data:
                    rate_info = season_data[rate_class]
                    rate_per_kwh = rate_info["tariffs"] + rate_info["eecc"]
                    usage = breakdown[season][day_type]["usage"][rate_class]
                    season_total_cost_precalc += usage * rate_per_kwh

        # Calculate $/day for season
        season_cost_per_day = (
            season_total_cost_precalc / total_days if total_days > 0 else 0.0
        )

        print(
            f"┌─ {season.upper()} (${season_cost_per_day:.2f}/day) "
            + "─" * (80 - len(season.upper()) - len(f"{season_cost_per_day:.2f}") - 11)
        )
        print("│")

        season_total_cost = 0.0

        # Process weekday and weekend/holiday
        for day_type in ["weekday", "weekend"]:
            day_type_data = breakdown[season][day_type]
            num_days = day_type_data["days"]

            if num_days == 0:
                continue

            # Calculate day type total cost
            day_type_cost = 0.0
            for rate_class in rates_classes:
                if rate_class in season_data:
                    rate_info = season_data[rate_class]
                    rate_per_kwh = rate_info["tariffs"] + rate_info["eecc"]
                    usage = day_type_data["usage"][rate_class]
                    day_type_cost += usage * rate_per_kwh

            # Calculate $/day
            cost_per_day = day_type_cost / num_days if num_days > 0 else 0.0

            # Print day type header
            day_type_label = "WEEKDAY" if day_type == "weekday" else "WEEKEND/HOLIDAY"
            print(
                f"│  ┌─ {day_type_label} (${cost_per_day:.2f}/day) "
                + "─" * (80 - len(day_type_label) - len(f"{cost_per_day:.2f}") - 16)
            )

            # Print rate classes
            day_type_kwh = 0.0
            for rate_class in rates_classes:
                if rate_class in season_data:
                    rate_info = season_data[rate_class]
                    rate_per_kwh = rate_info["tariffs"] + rate_info["eecc"]
                    usage = day_type_data["usage"][rate_class]
                    cost = usage * rate_per_kwh

                    # Format rate class name for display
                    rate_display = rate_class.replace("_", " ").title()

                    print(
                        f"│  │  {rate_display:20} {usage:>8.2f} kWh    ${rate_per_kwh:.4f}/kWh    ${cost:>8.2f}"
                    )
                    day_type_kwh += usage

            print(f"│  │                     {'─' * 7}                      {'─' * 7}")
            print(
                f"│  │  {day_type_label.capitalize():20} {day_type_kwh:>8.2f} kWh                  ${day_type_cost:>8.2f}"
            )
            print("│  └" + "─" * 74)
            print("│")

            season_total_cost += day_type_cost

        print(
            f"│  {season.capitalize()} Total:         {season_kwh:>8.2f} kWh                  ${season_total_cost:>8.2f}"
        )
        print("│")
        print("└" + "─" * 79)
        print()

        plan_subtotal += season_total_cost

        # Track baseline credit if present
        if "baseline_adjustment_credit" in season_data:
            # Will be calculated after subtotal
            pass

    # Calculate total baseline adjustment credit
    for season, season_data in plan_data.items():
        if season not in breakdown:
            continue
        if "baseline_adjustment_credit" in season_data:
            # Get season usage
            season_kwh = sum(
                [
                    breakdown[season][day_type]["usage"][rate_class]
                    for day_type in ["weekday", "weekend"]
                    for rate_class in rates_classes
                ]
            )
            # Get billing days
            billing_days = (
                breakdown[season]["weekday"]["days"]
                + breakdown[season]["weekend"]["days"]
            )
            # Get credit per kwh
            credit_per_kwh = -season_data["baseline_adjustment_credit"]

            # Calculate baseline deduction
            allowance_deduction = get_allowance_deduction(
                zone=zone,
                season=season,
                service_type=service_type,
                billing_days=billing_days,
                total_usage=season_kwh,
                credit_per_kwh=credit_per_kwh,
            )
            total_baseline_credit += allowance_deduction

    # Print totals
    print(
        f"Plan Subtotal:                                       ${plan_subtotal:>8.2f}"
    )
    print(
        f"Baseline Adjustment Credit:                          ${-total_baseline_credit:>8.2f}"
    )
    print("─" * 80)
    avg_rate = total_cost / total_usage if total_usage != 0 else 0.0
    print(
        f"PLAN TOTAL: {total_usage:.2f} kWh @ ${avg_rate:.4f}/kWh = ${total_cost:.2f}"
    )
    print()


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
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Show detailed breakdown by season, day type, and TOU period.",
)
def plot_sdge_hourly(filename, zone, pcia_year, solar, verbose):
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
    cca_schedules = os.path.join(pwd, "rates", "cca_schedules.yaml")
    pcia_file = os.path.join(pwd, "rates", "pcia.yaml")

    rates = load_yaml(sdge_schedules)
    cca_rates = load_yaml(cca_schedules)
    pcia_rates = load_yaml(pcia_file)

    # Generate CCA plans
    if cca_rates:
        cca_plans = generate_cca_plans(rates, cca_rates)
        rates.update(cca_plans)

    # Create list of all plans
    plans = list(rates.keys())

    c = SDGECaltulator(daily, rates, pcia_rates[int(pcia_year)], zone=zone, solar=solar)

    # Calculate charges for all plans first
    for plan in plans:
        estimated_charge = c.calculate(plan=plan)
        plans_and_charges[plan] = estimated_charge

    # Sort plans by cost
    sorted_plans = sorted(plans_and_charges.items(), key=lambda x: x[1])

    # Print detailed analysis in sorted order if verbose
    if verbose:
        for plan, estimated_charge in sorted_plans:
            plan_data = rates[plan]
            schedule = get_schedule_function(plan_data)
            breakdown = c.detailed_tally(schedule=schedule)
            print_detailed_analysis(
                plan,
                plan_data,
                breakdown,
                c.total_usage,
                estimated_charge,
                zone=zone,
                service_type="electric",
            )

    # Print summary section
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    c.print_info()
    print()

    for plan, charge in sorted_plans:
        print(f"{plan:<15} ${charge:.4f} ${charge / c.total_usage:.4f}/kWh")

    c.generate_plots()


if __name__ == "__main__":
    plot_sdge_hourly()
