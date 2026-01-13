# Self-Host Lipsync na Vast.ai – Guia Completo (Imagem + Áudio → Vídeo)

> **Objetivo:** Subir um serviço próprio que receba **1 imagem + 1 áudio** e devolva vídeo MP4 com lipsync, rodando numa GPU da Vast.ai (custo ~US$ 0,30/h) e pronto para ser chamado pelo **n8n**.

---

## 0) Pré-requisitos

- Conta grátis em [cloud.vast.ai](https://cloud.vast.ai) com cartão ou PayPal (para billing)
- 15 min livres
- (opcional) domínio/subdomínio no Cloudflare para túnel seguro

---

## 1) Escolha da GPU e Deploy Inicial

Quanto mais VRAM, mais rápido e maior resolução suportada. Para **InfiniteTalk int8 (19 GB)** ou **SadTalker (~6 GB)**:

| GPU | VRAM | Preço médio | Notas |
|-----|------|-------------|-------|
| RTX 4090 | 24 GB | **$0,29/h** | Roda 480p / 720p int8 sem swap |
| RTX 3090 | 24 GB | $0,35/h | Mesma performance, menos ofertas |
| A100 PCIe | 40 GB | $1,39/h | Ideal para 720p / múltiplos jobs |

### 1.1 Deploy via interface (clique rápido)

1. Acesse **“Create Instance”** → aba **Templates** → escolha **“NVIDIA CUDA”** (já vem driver 535 + CUDA 12.1 + Python 3.10).
2. **Launch Mode**: `SSH` (ou `Jupyter` se quiser notebook).
3. **Port Mapping**: clique em **+ Port** → Host `5000` → Container `5000` TCP.
4. **Disk Space**: Use pelo menos **160 GB** (ver nota na seção 3 sobre requisito de disco).
5. Clique **"Rent"** – o pod fica pronto em **~2 min**.
### 1.2 (Opcional) Deploy via CLI

```bash
pip install vastai
vastai login --api-key $VAST_API_KEY
vastai create instance \
  --image nvidia/cuda:12.1-devel-ubuntu22.04 \
  --gpu 1 --cpu 4 --ram 16 \
  --disk 160 --ports 5000:5000 \
  --on-start-cmd "apt update && apt install -y git ffmpeg python3-pip"
```

---

## 2) Entrar na máquina e preparar ambiente

```bash
# Conectar (use o IP/porta que o painel mostra)
ssh root@<IP> -p <PORTA>   # senha gerada automaticamente

# Verificar GPU
nvidia-smi            # deve mostrar GPU + driver 535
nvcc --version        # deve mostrar CUDA 12.1
```

---

## 3) Download RÁPIDO dos pesos

Tamanhos aproximados:

- **Wan2.1-I2V-14B-480P (base)** → 82,3 GB (obrigatório para o fluxo atual!)
- **InfiniteTalk single int8** → 19,5 GB
- **InfiniteTalk single (full)** → 9,95 GB (necessário para `--infinitetalk_dir`)
- **SadTalker** → ~1 GB

⚠️ **IMPORTANTE:** O comando `generate_infinitetalk.py` exige `--ckpt_dir weights/Wan2.1-I2V-14B-480P`, então o modelo **base Wan2.1 é OBRIGATÓRIO**. Não há caminho "sem Wan base" no fluxo documentado.

```bash
# Instalar hf CLI
pip install -U huggingface_hub
huggingface-cli login     # insira seu token com permissão de leitura

# ⚠️ REQUISITO DE DISCO:
# Wan2.1 base: 82,3 GB
# InfiniteTalk int8: 19,5 GB
# InfiniteTalk single: 9,95 GB
# Cache HF, deps, outputs temporários: ~30-50 GB
# TOTAL: ~140-180 GB recomendado (uso de 160-200 GB para segurança)

# Criar pastas
mkdir -p /workspace/weights /workspace/app
cd /workspace/weights

# Baixa o modelo int8 (19 GB) – suporta 480p em 24 GB VRAM
huggingface-cli download MeiGen-AI/InfiniteTalk \
  --local-dir . \
  --include "quant_models/*single_int8.*"

# Baixa o modelo single (9,95 GB) – NECESSÁRIO para --infinitetalk_dir
huggingface-cli download MeiGen-AI/InfiniteTalk \
  --local-dir . \
  --include "single/infinitetalk.safetensors"

# Baixa o modelo base Wan2.1 (82,3 GB) – OBRIGATÓRIO para --ckpt_dir
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir Wan2.1-I2V-14B-480P \
  --include "*.safetensors" "*.json" "*.pth"
```

*Dica: adicione `--resume-download` para continuar se o download cair.*

---

## 4) Clonar repositório + instalar dependências

```bash
cd /workspace
# Repo com código de lipsync (pode ser o oficial ou nosso wrapper)
git clone https://github.com/MeiGen-AI/InfiniteTalk.git app
cd app

# Drivers de áudio e vídeo
apt update && apt install -y ffmpeg libsndfile1

# PyTorch CUDA 12.1 + xformers (compatível com RTX 40xx)
pip install torch==2.4.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.28+cu121 --index-url https://download.pytorch.org/whl/cu121

# Restante das dependências
pip install -r requirements.txt  # do repo
pip install huggingface_hub accelerate einops librosa soundfile flask flask-cors
```

---

## 5) Criar API mínima (Flask)

Arquivo `/workspace/app/api.py`:

```python
import os, uuid, tempfile, subprocess, logging, json
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- config ----------
PORT = int(os.getenv("PORT", 5000))
TOKEN = os.getenv("API_TOKEN", "meu_token_facil")
RESOLUTION = os.getenv("RESOLUTION", "infinitetalk-480")  # 480 | 720
STEPS = int(os.getenv("STEPS", 40))
# -----------------------------

def wav216k(src, dst):
    """Converte qualquer áudio para 16 kHz mono wav"""
    subprocess.run([
        "ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", dst
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

@app.route("/lipsync", methods=["POST"])
def lipsync():
    # auth simples
    if request.headers.get("Authorization") != f"Bearer {TOKEN}":
        return {"error": "Unauthorized"}, 401

    # salva arquivos
    img_file = request.files["image"]
    aud_file = request.files["audio"]
    prompt = request.form.get("prompt", "A person talking.")

    tmp = tempfile.TemporaryDirectory()
    img_path = os.path.join(tmp.name, secure_filename(img_file.filename))
    aud_path = os.path.join(tmp.name, "audio_16k.wav")
    out_path = os.path.join(tmp.name, "output.mp4")

    img_file.save(img_path)
    # se o aud_file vier de um stream do Flask (request.files)
    with open(os.path.join(tmp.name, "input_audio"), "wb") as f:
        f.write(aud_file.read())
    wav216k(os.path.join(tmp.name, "input_audio"), aud_path)

    # gera vídeo (exemplo genérico – adapte para o modelo que baixou)
    logging.info("Iniciando geração...")
    subprocess.run([
        "python", "generate_infinitetalk.py",
        "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
        "--infinitetalk_dir", "weights/InfiniteTalk/single/infinitetalk.safetensors",
        "--quant_dir", "weights/InfiniteTalk/quant_models/infinitetalk_single_int8.safetensors",
        "--input_json", "/dev/stdin",
        "--save_file", out_path.replace(".mp4", ""),
        "--size", RESOLUTION,
        "--sample_steps", str(STEPS),
        "--mode", "clip",
        "--num_persistent_param_in_dit", "0"
    ], input=json.dumps({
        "prompt": prompt,
        "cond_video": img_path,
        "cond_audio": {"person1": aud_path}
    }), text=True, check=True)

    return send_file(out_path, mimetype="video/mp4",
                     as_attachment=True, download_name="lipsync.mp4")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
```

Inicie a API:

```bash
export API_TOKEN=meu_token_facil
export RESOLUTION=infinitetalk-480
export PORT=5000
python api.py
```

*Verifique logs: `* Running on all addresses (0.0.0.0) – http://0.0.0.0:5000`*

---

## 6) Expor para o mundo (3 opções)

### A) Direct (mais rápida)

- No painel da instância → **Networking** → garanta que `5000 TCP` esteja mapeado (feito no passo 1).
- Use a URL `http://<IP público>:<PORTA_MAPEADA>/lipsync` diretamente no n8n.

### B) Jupyter Proxy (se escolheu Launch Mode = Jupyter)

- Acesse `https://<id>.vast.ai/proxy/5000/lipsync` – não precisa abrir porta, mas é 1-2s mais lento.

### C) Cloudflare Tunnel (seguro + domínio próprio)

```bash
# Dentro do pod
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared
cloudflared tunnel --url http://localhost:5000
# Copie a URL https que aparecer (ex.: https://lipsync-abc123.cfargotunnel.com)
```

Aponte seu subdomínio (ex.: `lipsync.seudominio.com`) para esse túnel no painel do Cloudflare.

---

## 7) Teste rápido (antes de ir pro n8n)

```bash
# Shell local ou dentro do pod
curl -X POST http://localhost:5000/lipsync \
  -H "Authorization: Bearer meu_token_facil" \
  -F "image=@/caminho/foto.png" \
  -F "audio=@/caminho/audio.wav" \
  -F "prompt=A person talking to the camera" \
  --output video.mp4
```

Abra `video.mp4` – deve estar sincronizado.

---

## 8) Integração com n8n (exemplo de nó)

- **Tipo:** HTTP Request
- **Method:** POST
- **URL:** `http://<IP>:<PORTA>/lipsync` (ou URL do túnel)
- **Headers:**
  - `Authorization` = `Bearer meu_token_facil`
- **Body:** `multipart/form-data`
  - `image` → File (de upstream, ex. Google Drive)
  - `audio` → File (de upstream, ex. TTS)
  - `prompt` → String (opcional)

**O que vem a seguir?** (Veja a Seção 11 sobre armazenamento).

---

## 9) Troubleshooting comum

| Problema | Solução rápida |
|----------|----------------|
| `CUDA out of memory` | Use `--size infinitetalk-480` e `--num_persistent_param_in_dit 0` ou troque para RTX 4090 24 GB. |
| `Port 5000 refused` | Verifique **Networking** no painel → deve constar `5000 TCP` mapeado. |
| `ffmpeg not found` | `apt install -y ffmpeg` dentro do pod. |
| `cloudflared permission denied` | `chmod +x /usr/local/bin/cloudflared` ou baixar release correto para arquitetura (amd64). |

---

## 10) Custo & Tempo Resumido

| Etapa | Tempo real | Custo (RTX 4090) |
|-------|------------|------------------|
| Criar instância | 2 min | $0,29/h |
| Rodar script setup + downloads | 30-60 min¹ | ~$0,15-0,30 |
| Gerar 1 vídeo (30s) | ~45 s² | ~$0,004 |
| **TOTAL 1º vídeo** | **~35-65 min**¹ | **$0,15-0,35** |
| **Mensal** (1-2 vídeos/dia) | — | **~$1-2³** |

1. **Downloads**: Wan2.1 (82 GB) + InfiniteTalk (29 GB) leva 30-60 min dependendo da conexão.
2. **Tempo de vídeo**: Estimado para 30s em condições ideais. Para 1-4 min, faça benchmark real (veja seção 11.2).
3. **Custo mensal**: Considerando storage de ~120 GB (modelos + cache).

---

## 11) Armazenamento e Backup (MUITO IMPORTANTE!) 📦

> Pense na Vast.ai como um **"Fogão de Restaurantes"** e no seu n8n como o **"Entregador"**.
> - O fogão (GPU) prepara o prato (vídeo) e coloca no balcão (memória/disco da Vast.ai).
> - O entregador (n8n) leva o prato para a sua casa (Google Drive/YouTube).
> - **Se o entregador não pegar o prato e você desligar o fogão, o prato é jogado fora.**

### 11.1 Como a Vast.ai lida com arquivos

- **A Vast.ai NÃO é armazenamento permanente.**
- Quando você envia Imagem + Áudio, o servidor os salva num disco temporário (`/tmp` ou similar).
- O vídeo gerado também fica aí enquanto o servidor envia de volta.
- **Stop:** Pausa a instância mas **PRESERVA** os dados (continua cobrando storage).
- **Destroy:** Apaga permanentemente a instância e os dados.
- A próxima vez que você ligar, o sistema começa do zero (após Destroy).

### 11.2 Sobre Tempos de Processamento (Importante!)

Os números de tempo na tabela abaixo são **estimados otimistas** baseados em condições ideais. Para vídeos mais longos (1-4 minutos), o tempo **NÃO** escala linearmente. Recomendo fazer um benchmark real:

1. Teste com 30s, 60s, 120s, 240s de áudio
2. Meça o "tempo total wall clock" (inclui preprocess, inferência, encode)
3. Calcule o custo real: `custo = (tempo_em_horas * $/h) + custo_storage`

### 11.3 O que você DEVE fazer no n8n

Depois do nó **HTTP Request** que chama a Vast.ai (Seção 8), adicione **imediatamente** um nó de salvamento para que o vídeo não se perca. Veja as 3 opções principais:

#### A) Salvar no Google Drive (Recomendado) ☁️

1.  Adicione o nó **Google Drive**.
2.  **Operation:** Upload File.
3.  **Resource:** File.
4.  **Binary Property:** `data` (é o nome padrão da saída do nó anterior).
5.  **File Name (opcional):** `MarcoAurelio_video_{{ $now.format('dd-MM-yyyy_HH-mm') }}.mp4` (assim fica organizado).
6.  **Folder ID:** O ID da pasta no Drive onde quer salvar (ou deixe vazio para a raiz).

**Resultado:** O vídeo vai para o seu Google Drive e fica seguro para sempre.

#### B) Fazer upload direto para o YouTube (Para postar rápido) 📺

1.  Adicione o nó **YouTube**.
2.  **Operation:** Upload Video.
3.  **Binary Property:** `data`.
4.  **Title:** `Video Marco Aurélio - {{ $now.format('dd-MM-yyyy') }}`.
5.  **Privacy Status:** Public / Private / Unlisted.

**Resultado:** O vídeo sai da Vast.ai, passa pelo n8n e vai direto para o YouTube. Você não precisa baixar nada.

#### C) Salvar no disco do próprio n8n (Apenas para testes) 💾

1.  Adicione o nó **Write Binary File**.
2.  **File Name:** `/tmp/marco_video.mp4`.
3.  **Data:** `data`.

**Resultado:** O vídeo fica salvo na pasta temporária do servidor onde seu n8n roda.
**Cuidado:** Se você usar o n8n na nuvem (Docker, plano pago), esse espaço é limitado e pode ser difícil de acessar depois. **Use isso apenas para testes rápidos.**

### 11.4 Dicas de Segurança

- **Nunca confie na Vast.ai como único lugar dos seus vídeos.**
- Sempre tenha um nó de salvamento **Google Drive** ou **YouTube** antes do workflow acabar.
- Se o seu n8n der erro no meio, o vídeo some da Vast.ai. Tente usar nós de **Error Handling** para garantir que, mesmo com erro, você tente salvar o arquivo.

---

## 12) Comparativo de Custos: Vast.ai vs RunPod vs TensorDock (Para seu uso específico) 💰

Cenário Base: **45 vídeos por mês** (1,5 vídeo/dia), **40 s de áudio** cada, totalizando **~1,5 hora de GPU ativa/mês** (processamento + boot). Armazenamento incluído para **~25 GB** (Modelos Int8).

### Tabela Comparativa (RTX 4090 - 24 GB)

| Provedor | Preço/Hora (USD) | Custo Computação (1,5h) | Custo Armazenamento (120 GB/mês) | **TOTAL MENSAL (USD)** | **TOTAL MENSAL (BRL)** |
|-----------|---------------------|---------------------------|----------------------------------|--------------------------|------------------------|
| **Vast.ai** | **$0,29** | $0,44 | $6,00 | **$6,44** | **R$ 35,40** |
| **TensorDock** | **$0,37** | $0,56 | ~$5,00 | **$5,56** | **R$ 30,60** |
| **RunPod** | **$0,59** | $0,89 | ~$7,50 | **$8,39** | **R$ 46,10** |

*Nota: Storage de 120 GB inclui Wan2.1 (82 GB) + InfiniteTalk (29 GB) + cache. Câmbio estimado 1 USD = 5,5 BRL.*

*Nota: Câmbio estimado 1 USD = 5,5 BRL.*

### Análise de Custo-Benefício por Provedor

#### 1. Vast.ai (O Campeão de Mercado)
- **Custo:** $0,29/h.
- **Vantagens:** É o mais barato de longe. Mercado P2P enorme, então é fácil encontrar RTX 4090 barata.
- **Para você:** A economia é massiva. Mesmo somando o armazenamento de $1,83, é a opção mais barata para seu uso.
- **Quando usar:** Sempre que tiver RTX 4090 disponível. É a escolha padrão.

#### 2. TensorDock (A Surpresa)
- **Custo:** $0,37/h.
- **Vantagens:** Preço surpreendentemente competitivo, quase o mesmo nível da Vast.ai. Interface limpa e moderna.
- **Para você:** Na verdade, o **TensorDock é levemente mais barato que a Vast.ai** neste cenário (R$ 11,30 vs R$ 12,50).
- **Quando usar:** Excelente opção de **backup**. Se a Vast.ai estiver sem máquinas disponíveis, ou com preços altos, vá para a TensorDock.
- **Obs:** Tem menos máquinas disponíveis que a Vast.ai (mercado menor), então pode não ter GPU quando precisar.

#### 3. RunPod (O Profissional, mas Caro)
- **Custo:** $0,59/h.
- **Vantagens:** Interface e ferramentas muito melhores (Serverless, Networking fácil, Templates oficiais). Grande foco em produção (SOC 2, etc).
- **Para você:** Paga-se **~2x mais** que a Vast.ai. Difereça de **R$ 6,00 por mês** no seu bolso.
- **Quando usar:** Só se a Vast.ai e TensorDock estiverem instáveis demais para você. O custo não compensa para uso pessoal/manual.

### Conclusão: Qual escolher?

1.  **Principal:** **TensorDock** agora é levemente mais barato que Vast.ai neste cenário (R$ 30,60 vs R$ 35,40) devido ao custo de armazenamento menor.
2.  **Backup:** **Vast.ai**. Tem mais disponibilidade de GPUs e mercado maior.
3.  **Evitar:** **RunPod**. Para seu caso específico (1-2 vídeos/dia), é ~40% mais caro.

---

## Links Oficiais

- [Vast.ai Docs](https://docs.vast.ai/documentation/templates/creating-templates)
- [Vast.ai Pricing](https://docs.vast.ai/documentation/instances/pricing)
- [TensorDock Pricing](https://www.tensordock.com/gpu-4090.html)
- [RunPod Pricing](https://www.runpod.io/gpu-models/rtx-4090)
- [InfiniteTalk Weights](https://huggingface.co/MeiGen-AI/InfiniteTalk)
- [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)

Pronto! Seu endpoint `POST /lipsync` está no ar, e agora você tem a tabela de comparação detalhada para decidir onde gastar seus centavos. 🎬
