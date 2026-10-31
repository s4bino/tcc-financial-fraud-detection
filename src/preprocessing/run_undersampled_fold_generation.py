import undersampling

if __name__ == "__main__":
    # Os folds de partida são os de run_fold_generation.py, e val_size e random_state repetem os
    # usados lá na divisão interna, para que o treino interno subamostrado saia do mesmo treino
    # interno original.
    undersampling.undersampled_k_folds(
        csv_path="data/raw/creditcard.csv",
        folds_dir="data/processed/outer_folds",
        output_dir="data/undersampled",
        target="Class",
        target_legitimate=100_000,
        val_size=0.2,
        random_state=42,
    )
