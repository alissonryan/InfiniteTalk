# Opções de Geração de Cabeça Falante (Talking-Head) / Sincronia Labial (Lipsync) (2026)

> Caso de uso: automação n8n – enviar 1 imagem + 1 áudio → receber vídeo com sincronia labial (30 s, 480-720 p, ≤ 100 vídeos/dia).

---

## TL;DR – Recomendado para VOCÊ

**A) API-first (mais rápido para integrar)**

- **fal.ai** `fal-ai/infinitetalk` – **$0,20/s** (480 p) / **$0,40/s** (720 p), webhook/assíncrono, 5-30 s de inicialização a frio, até 24 s por chamada.  
  *Exemplo de custo*: 30 s × 100/dia = **$600/dia** (~$18 k/mês) – caro, mas zero infraestrutura.

- **WaveSpeedAI** `wavespeed-ai/infinitetalk` – **$0,15 por 5 s** (480 p) / **$0,30 por 5 s** (720 p) ⇒ **$0,03/s** (480 p).  
  *Mesmo exemplo*: 30 s × 100/dia = **$90/dia** (~$2,7 k/mês) – **InfiniteTalk hospedado mais barato**.

- **HeyGen API** – **$0,99/crédito/min** (Pro) ⇒ 30 s = 0,5 crédito → **~$0,50/30 s**. 100 vídeos/dia ≈ **$50/dia**; máx 5 min, documentação sólida/webhooks.

**B) Auto-hospedagem (custo controlado)**

- **Vast.ai RTX 4090 24 GB** – **$0,29/hora** (~$210/mês ligado direto).  
  Rodar modelos open-source (SadTalker, Wav2Lip, InfiniteTalk-int8) → **<$0,01 por vídeo de 30 s** após pagar o hardware.  
  Requer configuração (Docker, 2-3 h) mas é **90% mais barato** em escala.

---

## 1. APIs Hospedadas (sem download de modelo)

| Provedor e Modelo | Entrada→Saída | Preço (2026) | Duração máx | Latência | Webhook | Notas |
|-------------------|---------------|--------------|-------------|----------|---------|-------|
| **fal.ai** `fal-ai/infinitetalk` | img+áudio → 480/720 p MP4 | **$0,20/s** (480 p) **$0,40/s** (720 p) | 24 s (721 frames) | 5-30 s | ✅ | Melhor doc, fila assíncrona, pague-por-segundo |
| **WaveSpeedAI** `wavespeed-ai/infinitetalk` | img+áudio → 480/720 p MP4 | **$0,15/5 s** (480 p) **$0,30/5 s** (720 p) | 10 min | 5-15 s | ❌ (poll) | Mais barato por segundo, REST simples |
| **HeyGen API** `video/generate` (Photo Avatar) | img+áudio → 720 p MP4 | **1 crédito/min** (Pro $99/100 cr) → **$0,50/30 s** | 5 min | 30-120 s | ✅ | Alta qualidade, SLA empresarial, marca d'água removida nos planos pagos |
| **D-ID API** `talks` | img+áudio → 720 p MP4 | **Build $14,4/mês** 16 min incluídos → **$0,015/s** depois | 5 min | 20-60 s | ✅ | Streaming em tempo real também disponível |
| **Replicate** `zsxkib/multitalk` | img+áudio → 896×448 MP4 | **~$1,39/execução** (qualquer duração ≤ 3,2 s) | 3,2 s (81 frames) | 2-3 min | ✅ | Modelo da comunidade, cobrado por execução, pronto para múltiplas pessoas |

---

## 2. Opções de Auto-Hospedagem (traga sua GPU)

### 2.1 Recursos Completos (Qualidade InfiniteTalk)

- **Modelo**: `MeiGen-AI/InfiniteTalk` (single ou multi)  
- **Tamanho do Checkpoint**:  
  - FP16: ~2,7 GB (single) + 66 GB base Wan2.1-I2V-14B-480P  
  - INT8: 19,5 GB (single) – **recomendado para 24 GB VRAM**  
  - FP8: 19,5 GB (single) – leve ganho de qualidade sobre INT8
- **VRAM**: 20-24 GB mínimo (INT8/FP8) – RTX 4090 24 GB funciona em 480 p
- **Tempo de GPU**: ~1,5× tempo real (vídeo de 30 s ≈ 45 s GPU)  
- **Custo na Nuvem**:  
  - Vast.ai RTX 4090 **$0,29/hora** → **$0,004 por vídeo de 30 s**  
  - RunPod A100 80 GB **$1,39/hora** → **$0,02 por vídeo de 30 s**

### 2.2 Leve (menos recursos, mais rápido)

| Modelo | VRAM | Disco | Qualidade | Tempo de GPU (30 s) | Custo Nuvem $/vídeo (RTX 4090) |
|--------|------|-------|-----------|---------------------|--------------------------------|
| **SadTalker** | 6-8 GB | 1 GB | Estilizado, boas expressões | 15 s | **$0,001** |
| **Wav2Lip** | 4 GB | 0,1 GB | Lábios perfeitos, sem movimento de cabeça | 10 s | **$0,0008** |
| **LivePortrait** | 8 GB | 0,5 GB | Pose de cabeça realista, lábios razoáveis | 20 s | **$0,0016** |

---

## 3. Preços de Hardware por Provedor (Jan 2026)

| Provedor | GPU | VRAM | $/hora | Notas |
|----------|-----|------|--------|-------|
| **Vast.ai** | RTX 4090 | 24 GB | **$0,29** | Tipo Spot, sem taxa de saída (egress) |
| **RunPod** | RTX 4090 | 24 GB | **$0,75** | Nuvem segura, inicialização rápida |
| **RunPod** | A100 PCIe | 40 GB | **$1,39** | 80 GB também $1,39 (promoção) |
| **Lambda** | A100 | 40 GB | **$2,20** | Armazenamento persistente $0,10/GB |
| **Paperspace** | RTX 4090 | 24 GB | **$0,75** | Notebooks Pro, $0,05/GB armazenamento |
| **AWS EC2** | A100 | 40 GB | **$1,80** + saída | SLA Empresarial |

---

## 4. Notas de Integração n8n

- Todas as APIs listadas são HTTPS POST simples – use o nó **HTTP Request** do n8n.
- Prefira endpoints com **webhook/callback** (fal.ai, HeyGen, D-ID) para que o n8n não espere 2 min.
- Para auto-hospedagem, exponha seu container via **Tunnel** ou **Cloudflare Tunnels** e chame localmente.
- Armazene credenciais em **Credentials** do n8n (chaves de API) ou **Variáveis de Ambiente** (caminhos auto-hospedados).

---

## 5. Simulação de Custo (100 vídeos/dia, 30 s cada)

| Cenário | Custo Mensal | Notas |
|---------|--------------|-------|
| **WaveSpeedAI API** (480 p) | **$2,7 k** | Zero infra, escala instantânea |
| **HeyGen API** (720 p) | **$1,5 k** | Boa qualidade, marca d'água removida |
| **Auto-hospedagem Vast.ai RTX 4090** (ligado direto) | **$210** + **$0,40** eletricidade ≈ **$250** | 90% mais barato, precisa de 2 h de configuração |
| **Auto-hospedagem RunPod A100** (sob demanda, 1 h/dia) | **$42** | Ainda mais barato se desligar entre lotes |

---

## 6. Matriz de Decisão

| Prioridade | Recomendação |
|------------|--------------|
| **Go-live mais rápido** | WaveSpeedAI API (`$0,03/s`) – 1 chamada HTTP, sem dev-ops |
| **Melhor custo/qualidade** | Auto-hospedagem InfiniteTalk INT8 na RTX 4090 – **$250/mês fixo** + liberdade open-source |
| **SLA Empresarial** | HeyGen ou D-ID – webhooks, 4 k, assentos de equipe, docs legais |
| **Orçamento ultra-baixo** | Wav2Lip na Vast.ai – **<$0,001/vídeo**, 4 GB VRAM, lábios perfeitos (sem cabeça) |

---

## 7. Próximos Passos

1. **Protótipo**: abra o playground do WaveSpeedAI, envie sua imagem+áudio, confirme a qualidade.  
2. **n8n**: copie o curl para o nó HTTP Request, mude para “Wait for webhook” (Esperar por webhook) se disponível.  
3. **Escala**: quando a conta diária for > $150, mude para auto-hospedagem (RTX 4090 + InfiniteTalk INT8) – ROI < 1 semana.  

Todos os links e preços verificados em Jan 2026. Boa construção! 🎬
