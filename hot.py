import rawpy
import imageio
import numpy as np
import cv2
import os
import time
"""
Este código analisa os canais de cor RGB de uma imagem RAW para identificar discrepâncias causadas por aquecimento do sensor, conhecidas como hotpixels. Ele detecta esses pixels com base em desvios locais de cor, gera uma máscara que os marca visualmente, e aplica correções pontuais por interpolação.
Como resultado, você obterá três imagens:
- imagem_corrigida_final.png: imagem com os hotpixels removidos
- imagem_com_mascara.png: imagem original com os hotpixels destacados em vermelho
- mascara_raw.png: máscara em preto e branco mostrando os pixels detectados

"""
mostrar_mascara = True  # Ative ou desative a visualização da máscara

def detectar_hotpixels_por_desvio(imagem_rgb, limiar_contraste=60):
    h, w = imagem_rgb.shape[:2]
    mascara_total = np.zeros((h, w), dtype=np.uint8)

    r = imagem_rgb[:, :, 0].astype(np.int16)
    g = imagem_rgb[:, :, 1].astype(np.int16)
    b = imagem_rgb[:, :, 2].astype(np.int16)

    media_r = cv2.blur(r, (3, 3))
    media_g = cv2.blur(g, (3, 3))
    media_b = cv2.blur(b, (3, 3))

    diff_r = np.abs(r - media_r)
    diff_g = np.abs(g - media_g)
    diff_b = np.abs(b - media_b)

    mask_r = (diff_r > limiar_contraste).astype(np.uint8) * 255
    mask_g = (diff_g > limiar_contraste).astype(np.uint8) * 255
    mask_b = (diff_b > limiar_contraste).astype(np.uint8) * 255

    mascara_total = cv2.bitwise_or(mask_r, mask_g)
    mascara_total = cv2.bitwise_or(mascara_total, mask_b)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mascara_total = cv2.morphologyEx(mascara_total, cv2.MORPH_OPEN, kernel)

    return mascara_total

def corrigir_mascara(imagem, mascara):
    imagem_corrigida = imagem.copy()
    h, w = imagem.shape[:2]
    ys, xs = np.where(mascara == 255)

    for y, x in zip(ys, xs):
        if 1 < y < h - 2 and 1 < x < w - 2:
            vizinhos = imagem[y-1:y+2, x-1:x+2].reshape(-1, 3)
            imagem_corrigida[y, x] = np.median(vizinhos, axis=0)
    return imagem_corrigida

def aplicar_mascara_visual(imagem, mascara):
    """
    Sobrepõe a máscara como canal vermelho forte.
    """
    overlay = imagem.copy()
    vermelho = np.zeros_like(imagem)
    vermelho[:, :, 0] = mascara  # canal R
    alpha = 0.9  # opacidade mais forte
    overlay = cv2.addWeighted(overlay, 1.0, vermelho, alpha, 0)
    return overlay

def processar_dng():
    inicio = time.time()
    caminho_dng = r"seu caminho"
    print(f" Lendo imagem: {caminho_dng}")

    try:
        with rawpy.imread(caminho_dng) as raw:
            rgb16 = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=16
            )
    except Exception as e:
        print(f" Erro ao ler o DNG: {e}")
        return

    rgb = (rgb16 / 256).astype(np.uint8)

    print(" Detectando hotpixels por desvio local refinado")
    mascara = detectar_hotpixels_por_desvio(rgb, limiar_contraste=10) #limiar contraste muda a intensidade que esse filtro vai atuar na sua imagem 
    total = cv2.countNonZero(mascara)
    print(f" Hotpixels detectados: {total}")

    print(" Corrigindo pixels defeituosos")
    imagem_corrigida = corrigir_mascara(rgb, mascara)

    pasta_saida = os.path.join(os.getcwd(), 'corrigidas')
    os.makedirs(pasta_saida, exist_ok=True)

    if mostrar_mascara:
        print(" Gerando visualização da máscara...")
        imagem_com_mascara = aplicar_mascara_visual(rgb, mascara)
        caminho_mascara = os.path.join(pasta_saida, 'imagem_com_mascara.png')
        caminho_mask_raw = os.path.join(pasta_saida, 'mascara_raw.png')
        imageio.imwrite(caminho_mascara, imagem_com_mascara)
        imageio.imwrite(caminho_mask_raw, mascara)
        print(f" Imagem com máscara salva em: {caminho_mascara}")
        print(f" Máscara isolada salva em: {caminho_mask_raw}")

    caminho_saida = os.path.join(pasta_saida, 'imagem_corrigida_final.png')
    imageio.imwrite(caminho_saida, imagem_corrigida)
    print(f" Imagem corrigida salva em: {caminho_saida}")

    fim = time.time()
    print(f" Tempo total de processamento: {fim - inicio:.2f} segundos")

if __name__ == "__main__":
    print(" Iniciando processamento refinado com visualização...")
    processar_dng()
    print(" Finalizado.")
