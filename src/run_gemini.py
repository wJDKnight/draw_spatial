import os
import google.generativeai as genai
import pickle
from .utils import run_gemini, load_config
from .prompt import zeroshot_celltype_geneorder_grouped


genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-pro")
gen_config=genai.types.GenerationConfig(temperature=1.0, max_output_tokens=4000)

config = load_config("model_config/config_zeroshot_cosmx.yaml")

domain_text = "Inflammed Fibrosis,Airway,Lymphocyte Aggregate,Normal Alveoli,Inflammed Alveoli,Artery,Vein,Hyalinized Fibrosis"
if domain_text:
    domain_mapping = {i: domain_text.split(",")[i] for i in range(len(domain_text.split(",")))}
    config.domain_mapping = domain_mapping

print("Prompt example:")
print(zeroshot_celltype_geneorder_grouped(neighbor_celltype_df, neighbor_gene_df, rows=[0], config=config))

gemini_results_df, store_responses = run_gemini(model, gen_config, 
                                                neighbor_celltype_df, config, 
                                                zeroshot_celltype_geneorder_grouped, n_rows=1, 
                                                df_extra = neighbor_gene_df,
                                                column_name="zeroshot_gemini")

for i, cluster in enumerate(neighbor_celltype_df.index):
    print(f"Cluster{cluster}: {store_responses[i]}")


