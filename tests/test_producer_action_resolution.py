from src.core.tasks.producer_challenge.gameplay.decision import resolve_schedule_action_identity


def test_resolve_schedule_action_identity_maps_runtime_families():
    cases = [
        ("活動支給", "schedule_action_present_support", "present"),
        ("差し入れ", "schedule_action_fan_present", "present"),
        ("営業", "schedule_action_business", "business"),
        ("企業イベント", "schedule_action_business_corporate", "business"),
        ("自治体イベント", "schedule_action_business_municipal", "business"),
        ("リゾート施設", "schedule_action_business_resort", "business"),
        ("商業施設", "schedule_action_business_commercial", "business"),
        ("授業", "schedule_action_class", "activity"),
        ("おでかけ", "schedule_action_outing", "activity"),
        ("外出", "schedule_action_outing", "activity"),
        ("活動", "schedule_action_activity", "activity"),
        ("休む", "schedule_action_refresh", "refresh"),
        ("相談", "schedule_action_consult", ""),
    ]

    for title, expected_action_id, expected_rl_action_type in cases:
        resolution = resolve_schedule_action_identity(title, "unknown")
        assert resolution.action_id == expected_action_id
        assert resolution.metadata.get("rl_action_type", "") == expected_rl_action_type
        assert resolution.metadata.get("supported") is True


def test_resolve_schedule_action_identity_maps_manual_audition_variants():
    cases = [
        ("1次オーディション", "schedule_action_audition_first"),
        ("2次オーディション", "schedule_action_audition_second"),
        ("FINALE", "schedule_action_audition_finale"),
        ("特別オーディション", "schedule_action_audition"),
    ]

    for title, expected_action_id in cases:
        resolution = resolve_schedule_action_identity(title, "unknown")
        assert resolution.action_id == expected_action_id
        assert resolution.metadata.get("supported") is True


def test_resolve_schedule_action_identity_marks_customize_actions_as_todo():
    cases = [
        ("特別指導", "schedule_action_special_guidance"),
        ("カスタマイズ", "schedule_action_customize"),
    ]

    for title, expected_action_id in cases:
        resolution = resolve_schedule_action_identity(title, "unknown")
        assert resolution.action_id == expected_action_id
        assert resolution.source == "todo"
        assert resolution.metadata.get("supported") is False
        assert "TODO" in str(resolution.metadata.get("todo") or "")
