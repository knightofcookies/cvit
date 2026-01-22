class RealTextOCRDataset(Dataset):
    def __init__(self, tokenizer, num_samples=20000):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.fake = Faker('en_US')  # English generator

        self.transform = T.Compose([
            T.Resize((1024, 1024), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor()
        ]) # do not normalize, DeepSeek-OCR expects [0,1] range

    def __len__(self):
        return self.num_samples

    def generate_content(self):
        # Generate varied content types to make the model robust
        r = random.random()
        if r < 0.4:
            # Type 1: Standard Sentences (The easiest for LLMs)
            return self.fake.sentence(nb_words=10)
        elif r < 0.7:
            # Type 2: Addresses (Structured data)
            return self.fake.address().replace('\n', ', ')
        else:
            # Type 3: Names and Phone numbers
            return f"{self.fake.name()} - {self.fake.phone_number()}"

    def generate_image(self, text):
        # 1. Random Background (White-ish)
        bg_color = random.randint(230, 255)
        img = Image.new('RGB', (1024, 1024), color=(bg_color, bg_color, bg_color))
        draw = ImageDraw.Draw(img)

        # 2. Font Management
        try:
            # Attempt to use a larger, clearer font size
            font_size = random.randint(40, 80)
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # 3. Draw Text (Centered-ish)
        x = random.randint(50, 100)
        y = random.randint(200, 500)

        # Simple text wrapping logic
        words = text.split()
        current_line = ""
        for word in words:
            if (len(current_line) + len(word)) * (font_size * 0.5) > 800:
                draw.text((x, y), current_line, fill=(0, 0, 0), font=font)
                y += font_size + 10
                current_line = word + " "
            else:
                current_line += word + " "

        # Draw the last line
        draw.text((x, y), current_line, fill=(0, 0, 0), font=font)

        return img

    def __getitem__(self, idx):
        text = self.generate_content()
        image = self.generate_image(text)

        pixel_values = self.transform(image)
        # prompt = f"OCR: {text}{self.tokenizer.eos_token}"
        eos = self.tokenizer.eos_token if self.tokenizer.eos_token else "<|endoftext|>"
        prompt = f"OCR: {text}{eos}"

        # Consistent prefix masking logic from Stage 1 setup
        prefix_enc = self.tokenizer("OCR: ", add_special_tokens=False)
        prefix_len = len(prefix_enc.input_ids)

        encodings = self.tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings.input_ids.squeeze(0)
        attention_mask = encodings.attention_mask.squeeze(0)
        labels = input_ids.clone()

        # Mask prefix and padding
        starts_with_bos = (input_ids[0] == self.tokenizer.bos_token_id)
        offset = 1 if starts_with_bos else 0
        labels[:prefix_len + offset] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x['pixel_values'] for x in batch]),
        "input_ids": torch.stack([x['input_ids'] for x in batch]),
        "labels": torch.stack([x['labels'] for x in batch]),
        "attention_mask": torch.stack([x['attention_mask'] for x in batch])
    }

