"""Report every dependency in one pass; optional metadata never crashes initialization."""
import importlib.metadata as metadata
import importlib.util

DEPENDENCIES = [
    ('packaging','packaging','>=23',True),
    ('torch','torch','>=2.7,<2.10',True),('torchvision','torchvision','>=0.22,<0.25',True),
    ('numpy','numpy','>=1.24,<2',True),('Pillow','PIL','>=9',True),('PyYAML','yaml','>=6',True),
    ('pycocotools','pycocotools','>=2.0.7',True),('scipy','scipy','>=1.10',True),
    ('tqdm','tqdm','>=4.65',True),('matplotlib','matplotlib','>=3.7',True),
    ('filelock','filelock','>=3.12',True),('psutil','psutil','>=5.9',False),
    ('fvcore','fvcore','>=0.1.5',False),
]


def dependency_status():
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:
        try: from pip._vendor.packaging.specifiers import SpecifierSet
        except ImportError: SpecifierSet=None
    records=[]
    for package,module,spec,required in DEPENDENCIES:
        try: version=metadata.version(package)
        except metadata.PackageNotFoundError: version=None
        try: present=importlib.util.find_spec(module) is not None
        except (ImportError,ValueError): present=False
        ok=bool(present and version and SpecifierSet and version in SpecifierSet(spec))
        records.append(dict(package=package,module=module,required_version=spec,installed=version or '',
                            required_for_training=required,ok=ok))
    return records


def check_dependencies():
    records=dependency_status(); problems=[r for r in records if r['required_for_training'] and not r['ok']]
    if problems:
        lines='\n'.join(f"  {r['package']}{r['required_version']} | 当前：{r['installed'] or '未安装'}" for r in problems)
        print(f'[依赖提示] 以下包版本与建议范围不完全匹配（不阻止运行；若后续出现导入/数值错误请优先排查）：\n{lines}',flush=True)
    import torch,torchvision
    from packaging.version import Version
    tv=Version(torch.__version__.split('+')[0]); vv=Version(torchvision.__version__.split('+')[0])
    if tv.major != 2 or vv.major != 0 or vv.minor != tv.minor+15:
        print(f'[依赖提示] torch={torch.__version__} 与 torchvision={torchvision.__version__} 配对不标准，若报错请核对。',flush=True)
    return records
