# Limitless_Chat
<img src="https://github.com/Se-Boruk/Limitless_chat/blob/master/Assets/Limitless_logo.png?raw=true" alt="Limitless Logo" width="350"/>

This project aims to create an environment that allows you to use all the benefits of LLMs without concerns about privacy or answer refusal.

It is primarily an educational project intended for research, and I highly discourage using it in inappropriate ways, especially since, for now, it is only a skeleton prototype.

## Limitless_chat main points:
- No refusal policy
- Safe search
- Local
- Optimization


### No refusal policy:
Thanks to the use of abliterated models and special prompts, Limitless_chat is designed to answer every question. Of course, it can hallucinate and provide incorrect answers, but it is intended not to refuse any question.

### Safe search:
When searching for answers online, there is an option for Safe Search, which uses TOR to provide anonymity, though at the cost of speed.

### Local:
The project is designed to run as a local chat, providing an additional layer of anonymity.
The downside is that it requires using smaller models.

### Optimization:
Because of the point above, models are preferably run in quantized mode with optimization techniques (such as half-precision) so that the chat can be run with certain models on consumer-grade GPUs. The trade-off is slower “ready-to-action” responsiveness and a slight loss in accuracy.

## Layout
<img src="https://github.com/Se-Boruk/Limitless_chat/blob/master/Assets/Chat_preview.png?raw=true" alt="Limitless Logo" width="675"/>

## Due to the file sizes project does not contain
- Processed vector databse  
You must process it on your own with available docs (books, pdfs). 
- Models    
These must be downloaded separately from GitHub repo.


## Note
This project provides the tool for educational and research purposes. I cannot take responsibility for how it is used, so please use it responsibly.
