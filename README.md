# EMS1v scoring and ESM2 embeding calculation

Installation:  <code>conda env create -f env.yml</code>

## ESM1v scores:   
<code>conda activate esmenv</code>
<code>python quick_esm.py proteins.fasta --out-dir .</code>
   
## ESM2 embeddings:    
<code>conda activate esmenv</code>
<code> python esm2_emb_batch.py proteins.fasta --out-dir . --layer-agg last</code>

