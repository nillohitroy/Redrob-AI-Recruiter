import streamlit as st
import os
import subprocess
import pandas as pd

st.set_page_config(page_title="Redrob Ranker Sandbox", layout="wide")

st.title("Redrob AI Engineer Ranker - Sandbox Environment")
st.markdown("Upload a small `candidates.jsonl` sample. The system will run the headless `rank.py` pipeline using pre-computed embeddings and output the top candidates with LLM-generated reasoning.")

uploaded_file = st.file_uploader("Upload candidates.jsonl sample", type=['jsonl', 'json'])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    input_path = "temp_input.jsonl"
    output_path = "temp_output.csv"
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.info("File uploaded successfully. Initializing ranking pipeline...")
    
    if st.button("Run Ranker Engine"):
        with st.spinner("Running Semantic Recall, Behavioral Multipliers, and LLM Reasoning (Expect ~60 seconds for a small sample)..."):
            try:
                # We execute the required single command as a subprocess
                command = f"python rank.py --candidates {input_path} --out {output_path}"
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                
                if os.path.exists(output_path):
                    st.success("Ranking Complete!")
                    
                    # Display the data
                    df = pd.read_csv(output_path)
                    st.dataframe(df, use_container_width=True)
                    
                    # Provide download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download submission.csv",
                        data=csv,
                        file_name="team_submission.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("Pipeline failed to generate output.")
                    st.text("Subprocess Output Log:")
                    st.code(result.stderr)
                    
            except Exception as e:
                st.error(f"Execution failed: {str(e)}")