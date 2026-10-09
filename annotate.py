from ultralytics.data.annotator import auto_annotate
auto_annotate(data="images", det_model="yoloe-26m.seg.pt", sam_model="sam_b.pt", output_dir="dataset")
