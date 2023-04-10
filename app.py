import random

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

import torch
import os
import matplotlib.pyplot as plt
from sample_images_copy import main
app = Flask("flask-cifar")
i = 0
name = 1
protein_files = [
        "1bxn.x3d",
        "1c5u.x3d",
        "1cf5.x3d",
        "1e0p.x3d",
        "1f1b.x3d",
        "1hak.x3d",
        "1hwn.x3d",
        "1hwo.x3d",
        "1o6c.x3d",
        "1qoq.x3d",
        "1vim.x3d",
        "1viy.x3d",
        "1yg6.x3d",
        "1z30.x3d",
        "2ane.x3d",
        "2byu.x3d",
        "2fuv.x3d",
        "2h12.x3d",
        "2ldb.x3d",
        "2oqa.x3d",
        "2pcr.x3d",
        "2qes.x3d",
        "2wiz.x3d",
        "2z04.x3d",
        "2zzw.x3d",
        "3gl1.x3d",
        "3hhb.x3d",
        "3ku0.x3d",
        "3le7.x3d",
        "4d4r.x3d",
        "4dxy.x3d",
        "4fb9.x3d",
        "4leh.x3d",
        "4yov.x3d",
        "5ddz.x3d",
        "5wde.x3d",
        "5wvm.x3d",
        "5x1i.x3d",
        "5xls.x3d",
        "6afk.x3d",
        "6i7c.x3d",
        "6jaw.x3d",
        "6jhh.x3d",
        "6jhi.x3d",
        "6jvz.x3d",
        "6lov.x3d",
        "6loy.x3d",
        "6oo1.x3d",
        "6t3e.x3d",
        "7cgt.x3d"
    ]


protein_descriptions = [
    "1bxn - Bovine pancreatic trypsin inhibitor, a model system for protein folding and protein-protein interactions.",
    "1c5u - Influenza A virus hemagglutinin, a surface protein involved in viral entry into host cells.",
    "1cf5 - Human angiotensin-converting enzyme 2 (ACE2), the receptor for SARS-CoV-2 spike protein.",
    "1e0p - Murine coronavirus (MHV) spike protein receptor-binding domain, involved in viral attachment to host cells.",
    "1f1b - Human angiotensinogen, a precursor of angiotensin I and II involved in the regulation of blood pressure.",
    "1hak - Adenylate kinase, an enzyme involved in energy metabolism and homeostasis of cellular adenine nucleotide composition.",
    "1hwn - Heme-containing peroxidase, an enzyme that reduces hydrogen peroxide to water.",
    "1hwo - Heme-containing peroxidase complexed with a protein inhibitor.",
    "1o6c - SARS coronavirus main protease, an essential enzyme for viral replication.",
    "1qoq - Heat shock protein Hsp90, a molecular chaperone involved in protein folding and cellular stress response.",
    "1vim - Vibrio cholerae neuraminidase, an enzyme involved in the cleavage of sialic acid from host cell surface glycans.",
    "1viy - Vibrio cholerae neuraminidase complexed with an inhibitor.",
    "1yg6 - Human dihydroorotate dehydrogenase, an enzyme involved in the de novo synthesis of pyrimidines.",
    "1z30 - SARS coronavirus RNA-dependent RNA polymerase, an enzyme responsible for viral RNA synthesis.",
    "2ane - Human neutrophil elastase, a serine protease involved in inflammation and tissue remodeling.",
    "2byu - Human matrix metalloproteinase-2, an enzyme involved in the degradation of extracellular matrix components.",
    "2fuv - Human immunodeficiency virus type 1 (HIV-1) capsid protein, a structural protein essential for viral assembly.",
    "2h12 - Human transmembrane protease serine 2 (TMPRSS2), an enzyme that cleaves and activates the SARS-CoV-2 spike protein.",
    "2ldb - SARS coronavirus non-structural protein 3 (nsp3) papain-like protease domain, involved in viral polyprotein processing.",
    "2oqa - Bacteriophage T4 endonuclease VII, an enzyme involved in DNA recombination and repair.",
    "2pcr - Human procaspase-3, an inactive precursor of the apoptosis executioner enzyme caspase-3.",
    "2qes - Human histamine N-methyltransferase, an enzyme involved in the inactivation of histamine.",
    "2wiz - SARS coronavirus non-structural protein 3 (nsp3) macro domain, involved in ADP-ribose binding and potential roles in viral replication and host immune response modulation.",
    "2z04 - SARS coronavirus non-structural protein 5 (nsp5) main protease, an enzyme critical for the proteolytic processing of viral polyproteins.",
    "2zzw - Human cathepsin B, a lysosomal cysteine protease involved in protein degradation and processing.",
    "3gl1 - Influenza B virus neuraminidase, an enzyme involved in viral release from host cells.",
    "3hhb - Human hemoglobin, the oxygen-carrying protein in red blood cells.",
    "3ku0 - SARS coronavirus non-structural protein 9 (nsp9), a single-stranded RNA-binding protein involved in viral RNA replication.",
    "3le7 - Dengue virus non-structural protein 3 (NS3) helicase domain, an enzyme involved in viral RNA replication and unwinding.",
    "4d4r - SARS coronavirus non-structural protein 10 (nsp10), a cofactor for the viral RNA-dependent RNA polymerase and other replication enzymes.",
    "4dxy - Human angiotensin-converting enzyme (ACE), a peptidase involved in the regulation of blood pressure.",
    "4fb9 - Human cyclin-dependent kinase 2 (CDK2), an enzyme involved in cell cycle regulation.",
    "4leh - Human legumain, a lysosomal cysteine protease involved in protein degradation.",
    "4yov - Human beta-2 adrenergic receptor, a G protein-coupled receptor involved in physiological responses to adrenaline and noradrenaline.",
    "5ddz - Human macrophage migration inhibitory factor, a cytokine involved in immune cell regulation.",
    "5wde - SARS-CoV-2 spike protein receptor-binding domain, responsible for binding to the human ACE2 receptor.",
    "5wvm - SARS-CoV-2 spike protein receptor-binding domain complexed with a human ACE2 receptor.",
    "5x1i - Human CDK9/cyclin T1 complex, a key regulator of transcriptional elongation.",
    "5xls - SARS-CoV-2 spike protein, a surface protein that mediates viral entry into host cells.",
    "6afk - Human p38 MAP kinase, an enzyme involved in cellular stress response and inflammation.",
    "6i7c - SARS-CoV-2 non-structural protein 13 (nsp13) helicase, an enzyme involved in viral RNA replication and unwinding.",
    "6jaw - SARS-CoV-2 non-structural protein 15 (nsp15), an endoribonuclease involved in viral RNA processing.",
    "6jhh - SARS-CoV-2 non-structural protein 16 (nsp16), an enzyme involved in viral RNA capping.",
    "6jhi - SARS-CoV-2 non-structural protein 16 (nsp16) complexed with its cofactor nsp10.",
    "6jvz - SARS-CoV-2 RNA-dependent RNA polymerase, an enzyme responsible for viral RNA synthesis.",
    "6lov - SARS-CoV-2 main protease complexed with an inhibitor, a potential target for antiviral drug development.",
    "6loy - SARS-CoV-2 main protease in an unliganded form, an enzyme critical for the proteolytic processing of viral polyproteins.",
    "6oo1 - SARS-CoV-2 non-structural protein 3 (nsp3) papain-like protease domain, involved in viral polyprotein processing and immune evasion.",
    "6t3e - Human transmembrane protease serine 2 (TMPRSS2) in complex with a covalent inhibitor, an enzyme that cleaves and activates the SARS-CoV-2 spike protein.",
    "7cgt - Human cathepsin G, a neutrophil serine protease involved in inflammation and host defense."
]

protein_data = list(zip(protein_files, protein_descriptions))

count=0
@app.route("/", methods=["GET", "POST"])
def handle_request():
    global i
    global name
    protein_files_chunk = protein_data[i:i+6]
    label="1bxn"
    if request.method == "POST":
        
        data = dict(request.form)
        print(data)
        if "button_value" in data:
            button_value = data["button_value"]
            button_value = int(button_value)
            if button_value < 10:
                print(button_value)
                label=button_value
                check = True
                while check == True:
                    check = os.path.exists('./static/output/mrc/{}/'.format(name))
                    if check == True:
                        name = name + 1

                main(use_web=True, ids=name, LABEL=label)

                return render_template("test.html", name=name, label=label, protein_files_chunk=protein_files_chunk)
            
        if "next_row" in data:
            i += 6
            protein_files_chunk = protein_data[i:i+6]

        elif "previous_row" in data:
            i = max(i - 6, 0)
            protein_files_chunk = protein_data[i:i+6]

    return render_template('test.html', name=0, label="1bxn",protein_files_chunk=protein_files_chunk)

app.run(host="0.0.0.0", port=5010, debug=True)
