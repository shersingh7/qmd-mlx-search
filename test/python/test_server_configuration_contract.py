"""Cross-layer real-server configuration must match adapter capabilities."""
import inspect
from unittest.mock import patch
import pytest
from scripts import mlx_embed_server
from scripts.qmd_mlx.server import start_server


def test_server_defaults_use_qualified_model_and_infer_storage_quantization():
    defaults = inspect.signature(start_server).parameters
    assert defaults['model_name'].default == 'sentence-transformers/all-MiniLM-L6-v2'
    assert defaults['quantization'].default is None


@pytest.mark.parametrize('quant,expected', [('auto', None), ('none', 'none'), ('4bit', '4bit'), ('8bit', '8bit')])
def test_cli_passes_supported_quantization_to_runtime(monkeypatch, quant, expected):
    for name in ['MLX_EMBED_MODEL', 'MLX_EMBED_QUANT']:
        monkeypatch.delenv(name, raising=False)
    captured = {}
    def stop_before_loading(**kwargs):
        captured.update(kwargs)
        raise RuntimeError('configuration captured')
    with patch.object(mlx_embed_server, 'start_server', stop_before_loading), patch('sys.argv', ['mlx_embed_server.py', '--quantization', quant]):
        with pytest.raises(RuntimeError, match='configuration captured'):
            mlx_embed_server.main()
    assert captured['model_name'] == 'sentence-transformers/all-MiniLM-L6-v2'
    assert captured['quantization'] == expected
    assert captured['dtype_str'] == 'float32'
