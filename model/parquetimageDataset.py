class ParquetImageDataset(Dataset):
    def __init__(self, parquet_path, transform=None):
        self.df = pd.read_parquet(parquet_path)
        self.transform = transform
        self.df['Prompt'] = self.df['Prompt'].apply(self.clean_prompt)

    def clean_prompt(self, prompt):
        parts = prompt.split(',')
        return ', '.join([p.strip() for p in parts if not any(x in p.lower() for x in ['wearing', 'outfit', 'styled as'])])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_data = base64.b64decode(self.df.iloc[idx]['Image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        prompt = self.df.iloc[idx]['Prompt']
        tokens = tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_PROMPT_LENGTH).to(device)
        
        with torch.no_grad():
            outputs = embedder(**tokens)
            hidden_states = outputs.hidden_states[-1]
            prompt_embedding = hidden_states.mean(dim=1).squeeze(0)

        return image, prompt_embedding
