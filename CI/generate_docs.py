import yaml
from pathlib import Path

def yaml_to_markdown(yaml_file: Path, output_dir: Path):
    """Convert IDS YAML documentation to Markdown for MkDocs"""
    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    
    ids_name = yaml_file.stem
    md_lines = [
        f"# {ids_name.replace('_', ' ').title()}",
        "",
        data.get('system_overview', ''),
        "",
        "## Special Considerations",
        ""
    ]
    
    for item in data.get('special_considerations', []):
        md_lines.append(f"- {item}")
    
    md_lines.extend(["", "## IDS Entries", ""])
    
    for entry_path, entry_data in data.get('entries', {}).items():
        md_lines.extend([
            f"### `{entry_path}`",
            "",
            entry_data.get('summary', ''),
            "",
            entry_data.get('description', ''),
            ""
        ])
        
        if 'mds_path' in entry_data:
            md_lines.append(f"**MDSplus Path:** `{entry_data['mds_path']}`  ")
        if 'mds_paths' in entry_data:
            md_lines.append("**MDSplus Paths:**")
            for path in entry_data['mds_paths']:
                md_lines.append(f"- `{path}`")
        
        md_lines.append("")
    
    output_file = output_dir / f"{ids_name}.md"
    output_file.write_text('\n'.join(md_lines))

# Usage
docs_dir = Path('docs/ids')
docs_dir.mkdir(parents=True, exist_ok=True)

for yaml_file in Path('imas_composer/ids').glob('*.yaml'):
    yaml_to_markdown(yaml_file, docs_dir)
