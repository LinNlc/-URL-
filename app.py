"""Streamlit-based scheduling assistant application."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st


WEEKDAY_LABELS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


@dataclass
class Shift:
    name: str
    start_time: str
    end_time: str
    color: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} ({self.start_time}-{self.end_time})" if self.start_time and self.end_time else self.name


DEFAULT_SHIFTS: Dict[str, Shift] = {
    "白班": Shift("白班", "08:00", "16:00", "#fff7e6"),
    "中1": Shift("中1", "12:00", "20:00", "#e6f7ff"),
    "中2": Shift("中2", "14:00", "22:00", "#f9f0ff"),
    "夜班": Shift("夜班", "22:00", "08:00", "#f6ffed"),
}


def init_state() -> None:
    """Initialise Streamlit session state on first load."""
    if "groups" not in st.session_state:
        st.session_state.groups = {
            "A组": {"active": ["张三", "李四"], "archived": []},
            "B组": {"active": ["王五", "赵六"], "archived": []},
        }

    if "shift_definitions" not in st.session_state:
        st.session_state.shift_definitions = {name: shift for name, shift in DEFAULT_SHIFTS.items()}

    if "coverage_rules" not in st.session_state:
        st.session_state.coverage_rules: List[Dict[str, object]] = []

    if "schedule_meta" not in st.session_state:
        today = date.today()
        st.session_state.schedule_meta = {
            "start": date(today.year, today.month, 1),
            "months": 1,
        }

    if "schedule" not in st.session_state:
        st.session_state.schedule = build_schedule_dataframe(
            st.session_state.schedule_meta["start"],
            st.session_state.schedule_meta["months"],
        )

    if "summary_tolerance" not in st.session_state:
        st.session_state.summary_tolerance = 0

    if "selected_person" not in st.session_state:
        st.session_state.selected_person = ("", "")


def build_schedule_dataframe(start_date: date, months: int) -> pd.DataFrame:
    """Generate an empty schedule dataframe for the provided range."""
    start = pd.Timestamp(start_date).replace(day=1)
    end = (start + pd.offsets.MonthEnd(months - 1)).normalize()
    dates = pd.date_range(start=start, end=end, freq="D")

    groups: Dict[str, Dict[str, List[str]]] = st.session_state.groups
    columns: List[Tuple[str, str]] = [("日期", ""), ("星期", "")]
    for group_name, members in groups.items():
        for staff in members["active"]:
            columns.append((group_name, staff))

    df = pd.DataFrame(index=range(len(dates)), columns=pd.MultiIndex.from_tuples(columns), data="")
    df[("日期", "")] = dates
    df[("星期", "")] = [WEEKDAY_LABELS[ts.weekday()] for ts in dates]
    return df


def ensure_schedule_structure() -> None:
    """Ensure the schedule dataframe matches the current group/person structure."""
    schedule: pd.DataFrame = st.session_state.schedule
    expected_columns: List[Tuple[str, str]] = [("日期", ""), ("星期", "")]
    for group_name, members in st.session_state.groups.items():
        for staff in members["active"]:
            expected_columns.append((group_name, staff))

    expected_multi = pd.MultiIndex.from_tuples(expected_columns)
    if schedule is None or schedule.empty:
        st.session_state.schedule = build_schedule_dataframe(
            st.session_state.schedule_meta["start"],
            st.session_state.schedule_meta["months"],
        )
        return

    for column in expected_columns:
        if column not in schedule.columns:
            schedule[column] = ""

    drop_columns = [col for col in schedule.columns if col not in expected_columns]
    if drop_columns:
        schedule = schedule.drop(columns=drop_columns)

    schedule = schedule.reindex(columns=expected_multi)
    st.session_state.schedule = schedule


def regenerate_schedule(start_date: date, months: int) -> None:
    st.session_state.schedule_meta = {"start": start_date, "months": months}
    st.session_state.schedule = build_schedule_dataframe(start_date, months)


def sidebar_controls() -> None:
    with st.sidebar.expander("排班周期", expanded=True):
        current_start: date = st.session_state.schedule_meta["start"]
        current_months: int = st.session_state.schedule_meta["months"]
        start_date_input = st.date_input("开始月份", value=current_start, format="YYYY-MM-DD")
        start_date_input = start_date_input.replace(day=1)
        months_input = st.number_input("排班跨度（月）", min_value=1, max_value=12, value=current_months, step=1)
        if st.button("重新生成排班表", use_container_width=True):
            regenerate_schedule(start_date_input, int(months_input))
            st.success("已重新生成排班表，所有班次清空。")

    with st.sidebar.expander("组别与人员管理", expanded=True):
        with st.form("add_group_form", clear_on_submit=True):
            new_group = st.text_input("新组别名称")
            submitted = st.form_submit_button("添加组别")
            if submitted:
                if not new_group:
                    st.warning("请输入组别名称。")
                elif new_group in st.session_state.groups:
                    st.warning("该组别已存在。")
                else:
                    st.session_state.groups[new_group] = {"active": [], "archived": []}
                    ensure_schedule_structure()
                    st.success(f"已添加组别 {new_group}")

        if st.session_state.groups:
            with st.form("add_staff_form", clear_on_submit=True):
                group_names = list(st.session_state.groups.keys())
                selected_group = st.selectbox("选择组别", group_names)
                staff_name = st.text_input("人员姓名")
                submit_staff = st.form_submit_button("添加人员")
                if submit_staff:
                    if not staff_name:
                        st.warning("请输入人员姓名。")
                    elif staff_name in st.session_state.groups[selected_group]["active"]:
                        st.warning("该人员已存在于该组。")
                    else:
                        st.session_state.groups[selected_group]["active"].append(staff_name)
                        ensure_schedule_structure()
                        st.success(f"已向 {selected_group} 添加 {staff_name}")

        for group_name in list(st.session_state.groups.keys()):
            members = st.session_state.groups[group_name]
            st.markdown(f"**{group_name}**")
            active_members = members["active"]
            archived_members = members["archived"]
            if active_members:
                chosen_active = st.selectbox(
                    f"在 {group_name} 中操作的人员", active_members, key=f"active_{group_name}"
                )
                cols = st.columns(2)
                if cols[0].button("归档", key=f"archive_{group_name}"):
                    members["active"].remove(chosen_active)
                    members["archived"].append(chosen_active)
                    ensure_schedule_structure()
                    st.success(f"已归档 {chosen_active}")
                if cols[1].button("删除", key=f"delete_{group_name}"):
                    members["active"].remove(chosen_active)
                    ensure_schedule_structure()
                    st.success(f"已删除 {chosen_active}")
            else:
                st.caption("暂无在岗人员")

            if archived_members:
                chosen_archived = st.selectbox(
                    f"恢复 {group_name} 的归档人员", archived_members, key=f"archived_{group_name}"
                )
                if st.button("恢复", key=f"restore_{group_name}"):
                    members["archived"].remove(chosen_archived)
                    members["active"].append(chosen_archived)
                    ensure_schedule_structure()
                    st.success(f"已恢复 {chosen_archived}")

            if st.button("删除组别", key=f"remove_group_{group_name}"):
                del st.session_state.groups[group_name]
                ensure_schedule_structure()
                st.warning(f"已删除组别 {group_name}")
                st.experimental_rerun()

    with st.sidebar.expander("班次管理", expanded=False):
        if st.session_state.shift_definitions:
            st.markdown("当前班次：")
            for name, shift in st.session_state.shift_definitions.items():
                st.markdown(f"- **{name}**：{shift.start_time}-{shift.end_time}")

        with st.form("add_shift_form", clear_on_submit=True):
            shift_name = st.text_input("班次名称")
            col_a, col_b = st.columns(2)
            start_time = col_a.text_input("开始时间", placeholder="08:00")
            end_time = col_b.text_input("结束时间", placeholder="16:00")
            color = st.color_picker("显示颜色", value="#FFFFFF")
            add_shift = st.form_submit_button("新增/覆盖班次")
            if add_shift:
                if not shift_name:
                    st.warning("请输入班次名称。")
                else:
                    st.session_state.shift_definitions[shift_name] = Shift(
                        shift_name, start_time, end_time, color
                    )
                    st.success(f"已保存班次 {shift_name}")

        if st.session_state.shift_definitions:
            shift_to_remove = st.selectbox(
                "删除班次", [""] + list(st.session_state.shift_definitions.keys()), key="remove_shift"
            )
            if shift_to_remove and st.button("确认删除班次"):
                st.session_state.shift_definitions.pop(shift_to_remove, None)
                st.success(f"已删除班次 {shift_to_remove}")

    with st.sidebar.expander("关键班次提醒", expanded=False):
        if st.session_state.shift_definitions:
            with st.form("coverage_rule_form", clear_on_submit=True):
                shift_options = list(st.session_state.shift_definitions.keys())
                selected_shift = st.selectbox("需要覆盖的班次", shift_options)
                default_start = st.session_state.schedule_meta["start"]
                default_end = (
                    default_start
                    + pd.offsets.MonthEnd(st.session_state.schedule_meta["months"] - 1)
                ).date()
                col1, col2 = st.columns(2)
                start_required = col1.date_input("开始日期", value=default_start)
                end_required = col2.date_input("结束日期", value=default_end)
                description = st.text_input("备注（可选）")
                submitted_rule = st.form_submit_button("添加提醒")
                if submitted_rule:
                    if start_required > end_required:
                        st.warning("开始日期不能晚于结束日期。")
                    else:
                        st.session_state.coverage_rules.append(
                            {
                                "shift": selected_shift,
                                "start": start_required,
                                "end": end_required,
                                "description": description,
                            }
                        )
                        st.success("提醒已添加。")

        if st.session_state.coverage_rules:
            for idx, rule in enumerate(st.session_state.coverage_rules):
                desc = rule["description"] or ""
                st.write(
                    f"{idx + 1}. {rule['shift']} {rule['start']} - {rule['end']} {desc}"
                )
                if st.button("移除此提醒", key=f"remove_rule_{idx}"):
                    st.session_state.coverage_rules.pop(idx)
                    st.experimental_rerun()


def export_tools(schedule: pd.DataFrame) -> None:
    """导出排班与汇总数据为 Excel 供线下归档。"""

    display_df = schedule.copy()
    display_df[("日期", "")] = pd.to_datetime(display_df[("日期", "")]).dt.strftime("%Y-%m-%d")

    summary_df = build_summary(schedule)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        display_df.to_excel(writer, sheet_name="排班表", index=False)
        if not summary_df.empty:
            summary_df.reset_index().to_excel(writer, sheet_name="统计汇总", index=False)

    st.download_button(
        "导出为Excel",
        data=output.getvalue(),
        file_name=f"排班表_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def flatten_schedule_for_editor(schedule: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, str]]]:
    """Flatten multi-level columns to string labels for editing."""
    label_map: Dict[str, Tuple[str, str]] = {}
    flat_columns: List[str] = []
    for top, bottom in schedule.columns:
        if top in ("日期", "星期"):
            label = top
        else:
            label = f"{top}｜{bottom}"
        label_map[label] = (top, bottom)
        flat_columns.append(label)

    flat_df = schedule.copy()
    flat_df.columns = flat_columns
    if "日期" in flat_df.columns:
        flat_df["日期"] = pd.to_datetime(flat_df["日期"]).dt.strftime("%Y-%m-%d")
    return flat_df, label_map


def update_schedule_from_editor(edited: pd.DataFrame, label_map: Dict[str, Tuple[str, str]]) -> None:
    schedule: pd.DataFrame = st.session_state.schedule
    for label, (top, bottom) in label_map.items():
        if label == "日期":
            schedule[("日期", "")] = pd.to_datetime(edited[label])
        elif label == "星期":
            schedule[("星期", "")] = edited[label]
        else:
            schedule[(top, bottom)] = edited[label].fillna("")
    st.session_state.schedule = schedule


def schedule_editor() -> None:
    schedule: pd.DataFrame = st.session_state.schedule
    flat_df, label_map = flatten_schedule_for_editor(schedule)

    shift_options = [""] + list(st.session_state.shift_definitions.keys())
    column_config = {
        "日期": st.column_config.Column("日期", disabled=True),
        "星期": st.column_config.Column("星期", disabled=True),
    }
    shift_labels = " / ".join(
        shift.label for shift in st.session_state.shift_definitions.values()
    )

    for label in flat_df.columns:
        if label in ("日期", "星期"):
            continue
        column_config[label] = st.column_config.SelectboxColumn(
            label,
            options=shift_options,
            default="",
            help="选择班次，留空表示未排班。可选班次：" + shift_labels,
        )

    edited = st.data_editor(
        flat_df,
        column_config=column_config,
        use_container_width=True,
        height=500,
        key="schedule_editor",
    )
    update_schedule_from_editor(edited, label_map)


def build_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    shift_names = list(st.session_state.shift_definitions.keys())
    person_columns = [col for col in schedule.columns if col[0] not in ("日期", "星期")]
    summary_records: List[Dict[str, object]] = []

    for group, staff in person_columns:
        column_series = schedule[(group, staff)]
        record = {"组别": group, "姓名": staff}
        non_empty = column_series.replace("", pd.NA).dropna()
        record["总班次"] = int(non_empty.shape[0])
        for shift_name in shift_names:
            record[shift_name] = int((column_series == shift_name).sum())
        summary_records.append(record)

    if not summary_records:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df.set_index(["组别", "姓名"])
    return summary_df


def highlight_summary(summary_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    if summary_df.empty:
        return summary_df.style

    shift_columns = [col for col in summary_df.columns if col != "总班次"]
    averages = summary_df[shift_columns].mean()
    tolerance = st.session_state.get("summary_tolerance", 0)

    def highlight(cell, column):
        if pd.isna(cell):
            return ""
        if cell > averages[column] + tolerance:
            return "background-color: #ffd6d6; font-weight: bold;"
        return ""

    styler = summary_df.style
    for column in shift_columns:
        styler = styler.applymap(lambda val, col=column: highlight(val, col), subset=pd.IndexSlice[:, column])
    return styler


def filter_schedule_by_range(schedule: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    mask = (schedule[("日期", "")] >= pd.Timestamp(start)) & (
        schedule[("日期", "")] <= pd.Timestamp(end)
    )
    return schedule.loc[mask]


def coverage_analysis(schedule: pd.DataFrame) -> List[str]:
    messages: List[str] = []
    person_columns = [col for col in schedule.columns if col[0] not in ("日期", "星期")]

    for rule in st.session_state.coverage_rules:
        mask = (
            schedule[("日期", "")] >= pd.Timestamp(rule["start"])
        ) & (schedule[("日期", "")] <= pd.Timestamp(rule["end"]))
        if not mask.any():
            continue
        sub = schedule.loc[mask, person_columns]
        dates = schedule.loc[mask, ("日期", "")]
        missing_dates = []
        assignment_matrix = sub.eq(rule["shift"]) if not sub.empty else pd.DataFrame(index=sub.index)
        has_shift_series = (
            assignment_matrix.any(axis=1)
            if not assignment_matrix.empty
            else pd.Series(False, index=sub.index, dtype=bool)
        )
        for current_date, has_shift in zip(dates, has_shift_series):
            if not bool(has_shift):
                missing_dates.append(current_date.date())
        if missing_dates:
            details = ", ".join(str(d) for d in missing_dates)
            desc = f"（{rule['description']}）" if rule.get("description") else ""
            messages.append(
                f"{rule['shift']} 在 {rule['start']} - {rule['end']} 内缺少排班日期：{details}{desc}"
            )
    return messages


def styled_schedule(schedule: pd.DataFrame) -> pd.io.formats.style.Styler:
    display_df = schedule.copy()
    display_df[("日期", "")] = pd.to_datetime(display_df[("日期", "")]).dt.strftime("%Y-%m-%d")
    shift_colors = {name: shift.color for name, shift in st.session_state.shift_definitions.items() if shift.color}
    person_columns = [col for col in display_df.columns if col[0] not in ("日期", "星期")]

    def highlight_cell(val: str) -> str:
        if val in shift_colors:
            return f"background-color: {shift_colors[val]};"
        return ""

    styler = display_df.style.applymap(highlight_cell, subset=pd.IndexSlice[:, person_columns])
    return styler


def statistics_section(schedule: pd.DataFrame) -> None:
    st.subheader("统计分析")
    summary_df = build_summary(schedule)
    if summary_df.empty:
        st.info("请先维护人员并排班，以查看统计数据。")
        return

    with st.expander("偏差阈值设置", expanded=False):
        st.caption("超过平均值多少次后标红，可用来控制容忍度。")
        shift_only = summary_df.drop(columns="总班次", errors="ignore")
        candidates = shift_only.to_numpy().flatten().tolist() if not shift_only.empty else []
        max_shift = int(max(candidates + [5]))
        tolerance = st.slider(
            "超出平均值（班次）阈值",
            min_value=0,
            max_value=max_shift,
            value=min(st.session_state.summary_tolerance, max_shift),
        )
        st.session_state.summary_tolerance = tolerance

    st.markdown("**整体概览**")
    st.write(highlight_summary(summary_df))

    st.markdown("**时间段统计**")
    overall_start = st.session_state.schedule_meta["start"]
    overall_end = (
        overall_start + pd.offsets.MonthEnd(st.session_state.schedule_meta["months"] - 1)
    ).date()
    col1, col2 = st.columns(2)
    range_start = col1.date_input("开始日期", value=overall_start)
    range_end = col2.date_input("结束日期", value=overall_end)
    if range_start > range_end:
        st.warning("开始日期不能晚于结束日期。")
    else:
        filtered = filter_schedule_by_range(schedule, range_start, range_end)
        range_summary = build_summary(filtered)
        if range_summary.empty:
            st.info("选定时间段内没有排班记录。")
        else:
            st.write(highlight_summary(range_summary))

    st.markdown("**按月份统计**")
    month_series = pd.to_datetime(schedule[("日期", "")]).dt.to_period("M").astype(str)
    month_options = month_series.unique().tolist()
    selected_month = st.selectbox("选择月份", month_options)
    month_df = schedule.loc[month_series == selected_month]
    month_summary = build_summary(month_df)
    st.write(highlight_summary(month_summary))

    person_columns = [col for col in schedule.columns if col[0] not in ("日期", "星期")]
    if person_columns:
        st.markdown("**个人详情**")
        groups = sorted({group for group, _ in person_columns})
        group_choice = st.selectbox("选择组别", [""] + groups, index=groups.index(st.session_state.selected_person[0]) + 1 if st.session_state.selected_person[0] in groups else 0)
        people = [name for grp, name in person_columns if grp == group_choice] if group_choice else []
        person_choice = st.selectbox(
            "选择人员",
            [""] + people,
            index=people.index(st.session_state.selected_person[1]) + 1 if group_choice and st.session_state.selected_person[1] in people else 0,
        )
        st.session_state.selected_person = (group_choice, person_choice)

        if group_choice and person_choice:
            person_series = schedule[(group_choice, person_choice)]
            person_df = pd.DataFrame({"日期": schedule[("日期", "")], "班次": person_series})
            person_df = person_df[person_df["班次"] != ""]
            if person_df.empty:
                st.info("该人员在所选周期内暂无排班记录。")
            else:
                monthly = (
                    person_df.assign(月份=pd.to_datetime(person_df["日期"]).dt.to_period("M"))
                    .groupby(["月份", "班次"])
                    .size()
                    .unstack(fill_value=0)
                )
                st.dataframe(monthly, use_container_width=True)

                shift_counts = person_df["班次"].value_counts().rename("次数").to_frame()
                st.dataframe(shift_counts, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="灵活排班助手", layout="wide")
    init_state()
    ensure_schedule_structure()

    st.title("灵活排班助手")
    st.caption("以表格化视图提升排班效率，支持自定义班次、人员及提醒逻辑。")

    sidebar_controls()

    st.subheader("常用工具")
    export_tools(st.session_state.schedule)

    st.subheader("排班表编辑")
    st.info("双击单元格即可选择班次，支持自定义班次名称与颜色。")
    schedule_editor()

    st.subheader("排班表预览")
    st.write(styled_schedule(st.session_state.schedule))

    warnings = coverage_analysis(st.session_state.schedule)
    if warnings:
        st.error("\n".join(warnings))
    else:
        st.success("关键班次覆盖情况良好。")

    statistics_section(st.session_state.schedule)

    st.markdown("---")
    st.caption(
        "提示：可以通过侧边栏维护人员、班次与提醒逻辑，所有数据保存在当前会话中。"
    )


if __name__ == "__main__":
    main()
