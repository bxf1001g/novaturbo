"""
Tests for the material properties module.
"""

import pytest

from src.physics.materials import (
    load_materials,
    check_thermal_limits,
    recommend_material,
    DEFAULT_MATERIALS,
    MaterialProperties,
)


def test_default_materials_non_empty():
    """DEFAULT_MATERIALS must contain at least the four core alloys."""
    for key in ('inconel_718', 'inconel_625', 'ti6al4v', 'ss316l'):
        assert key in DEFAULT_MATERIALS


def test_load_materials_returns_dict():
    """load_materials() should return a non-empty dict."""
    mats = load_materials()
    assert isinstance(mats, dict)
    assert len(mats) > 0


def test_load_materials_auto_detects_yaml():
    """Auto-detected YAML should include at least the core alloys."""
    mats = load_materials()
    for key in ('inconel_718', 'ti6al4v', 'ss316l'):
        assert key in mats, f"Expected '{key}' in loaded materials"


def test_load_materials_yaml_richer_than_defaults():
    """YAML database should have more (or equal) entries than the hardcoded defaults."""
    mats_yaml = load_materials()
    assert len(mats_yaml) >= len(DEFAULT_MATERIALS)


def test_material_properties_types():
    """All material property fields should be populated with correct types."""
    mat = DEFAULT_MATERIALS['inconel_718']
    assert isinstance(mat.name, str) and mat.name
    assert mat.density_kg_m3 > 0
    assert mat.max_service_temp_K > 0
    assert mat.yield_strength_MPa > 0
    assert isinstance(mat.printable_dmls, bool)


def test_check_thermal_limits_within_range():
    """Operating below max_service_temp should return valid=True with status OK."""
    result = check_thermal_limits('ti6al4v', 500)  # below 673 K limit
    assert result['valid'] is True
    assert result['status'] == 'OK'
    assert result['margin_continuous_K'] > 0


def test_check_thermal_limits_short_term():
    """Operating between continuous and short-term limits should return SHORT_TERM_ONLY."""
    # Inconel 718: max_service=973 K, max_short_term=1253 K
    result = check_thermal_limits('inconel_718', 1100)
    assert result['valid'] is False
    assert result['status'] == 'SHORT_TERM_ONLY'
    assert result['margin_short_term_K'] > 0


def test_check_thermal_limits_exceeds():
    """Operating above both limits should return EXCEEDS_LIMITS."""
    result = check_thermal_limits('ti6al4v', 900)  # well above 773 K short-term
    assert result['valid'] is False
    assert result['status'] == 'EXCEEDS_LIMITS'


def test_check_thermal_limits_unknown_material():
    """Unknown material key should return valid=False with an error field."""
    result = check_thermal_limits('unobtainium', 1000)
    assert result['valid'] is False
    assert 'error' in result


def test_recommend_material_returns_valid_key():
    """recommend_material should return a key present in the materials dict."""
    mats = load_materials()
    key = recommend_material(600, 'compressor', mats)
    assert key in mats


def test_recommend_material_respects_temperature():
    """Recommended material must be able to handle the given operating temperature."""
    mats = load_materials()
    temp = 400
    key = recommend_material(temp, 'compressor', mats)
    mat = mats[key]
    assert mat.max_service_temp_K >= temp


def test_recommend_material_prefers_cheaper():
    """All else being equal, recommend_material should prefer cheaper materials."""
    mats = load_materials()
    # At low temperature, SS316L (cheapest) should be preferred over Inconel
    key_low = recommend_material(300, 'housing', mats)
    assert mats[key_low].cost_per_kg_usd <= mats['inconel_718'].cost_per_kg_usd


def test_check_thermal_limits_uses_loaded_materials():
    """check_thermal_limits should work with the auto-loaded YAML materials."""
    mats = load_materials()
    result = check_thermal_limits('inconel_718', 900, mats)
    assert result['valid'] is True
