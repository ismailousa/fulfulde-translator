import xml.etree.ElementTree as ET
import json

def tmx_to_jsonl(tmx_path, jsonl_path, lang_src="en", lang_tgt="fuv"):
    tree = ET.parse(tmx_path)
    root = tree.getroot()

    # TMX structure: <tmx><body><tu>...</tu></body></tmx>
    body = root.find('body')

    with open(jsonl_path, 'w', encoding='utf-8') as out_file:
        for tu in body.findall('tu'):
            seg_src = None
            seg_tgt = None
            for tuv in tu.findall('tuv'):
                lang = tuv.attrib.get('{http://www.w3.org/XML/1998/namespace}lang')
                seg = tuv.find('seg').text if tuv.find('seg') is not None else None
                if lang == lang_src:
                    seg_src = seg
                elif lang == lang_tgt:
                    seg_tgt = seg
            if seg_src and seg_tgt:
                json_line = json.dumps({"en": seg_src, "ff": seg_tgt}, ensure_ascii=False)
                out_file.write(json_line + '\n')

if __name__ == "__main__":
    # Example usage:
    tmx_path = "data/all.en-fuv.tmx"
    jsonl_path = "data/adamawa_covid_english_fulfulde.jsonl"
    tmx_to_jsonl(tmx_path, jsonl_path)
    print(f"Converted TMX to JSONL saved at: {jsonl_path}")
