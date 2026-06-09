import matplotlib.pyplot as plt
import numpy as np

# --- DATOS DEL ANEXO (3 Intentos por Experimento) ---

# Experimento 1: Meta y Estado Inicial Aleatorios
exp1_her_data = {
    2: [76.60, 73.60, 77.40],
    10: [87.60, 90.80, 85.20],
    20: [87.20, 88.00, 87.40],
    30: [68.40, 68.60, 68.80],
    50: [0.40, 0.40, 0.05]
}
exp1_dqn_data = {
    2: [75.00, 77.60, 76.40],
    5: [90.59, 90.90, 91.08],
    10: [59.13, 62.85, 56.79],
    15: [0.01, 0.00, 0.00],
    20: [0.00, 0.00, 0.01]
}

# Experimento 2: Meta Fija, Estado Inicial Aleatorio
exp2_her_data = {
    2: [73.28, 73.00, 73.61],
    5: [90.43, 90.37, 90.21],
    10: [84.44, 83.97, 84.15],
    15: [80.13, 80.20, 80.41],
    20: [72.52, 71.20, 71.08],
    30: [37.33, 36.23, 38.24],
    50: [0.06, 0.06, 0.00]
}
exp2_dqn_data = {
    2: [72.81, 72.71, 72.37],
    5: [90.78, 91.00, 90.62],
    10: [54.60, 55.24, 66.54],
    15: [0.01, 0.03, 0.02],
    20: [0.00, 0.00, 0.00]
}

# Experimento 3: Meta Fija, Estado Inicial Opuesto
exp3_her_data = {
    2: [97.57, 97.49, 97.44],
    5: [79.92, 80.44, 80.39],
    10: [63.70, 64.21, 63.83],
    15: [67.83, 68.70, 68.33],
    20: [53.84, 52.98, 54.20],
    30: [9.58, 11.80, 10.78],
    50: [0.00, 0.00, 0.00]
}
exp3_dqn_data = {
    2: [97.47, 97.61, 97.50],
    5: [92.53, 92.44, 92.50],
    10: [0.01, 0.01, 0.01],
    15: [0.00, 0.00, 0.00],
    20: [0.00, 0.00, 0.00]
}

def plot_and_save_variance(dqn_dict, her_dict, title, filename):
    plt.figure(figsize=(10, 6))
    
    # Procesar y pintar HER (Azul)
    her_x = sorted(her_dict.keys())
    her_means = [np.mean(her_dict[b]) for b in her_x]
    her_stds = [np.std(her_dict[b]) for b in her_x]
    plt.plot(her_x, her_means, 'bo-', label='HER (Media)', linewidth=2)
    plt.fill_between(her_x, np.array(her_means) - np.array(her_stds), 
                     np.array(her_means) + np.array(her_stds), color='blue', alpha=0.15)
    
    # Procesar y pintar DQN (Rojo)
    dqn_x = sorted(dqn_dict.keys())
    dqn_means = [np.mean(dqn_dict[b]) for b in dqn_x]
    dqn_stds = [np.std(dqn_dict[b]) for b in dqn_x]
    plt.plot(dqn_x, dqn_means, 'ro-', label='DQN (Media)', linewidth=2)
    plt.fill_between(dqn_x, np.array(dqn_means) - np.array(dqn_stds), 
                     np.array(dqn_means) + np.array(dqn_stds), color='red', alpha=0.15)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Número de bits ($n$)', fontsize=12)
    plt.ylabel('Tasa de Éxito Óptimo (%)', fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Guardar imagen
    plt.savefig(filename)
    print(f"Imagen guardada: {filename}")
    plt.close()

# Ejecutar la creación de gráficas
plot_and_save_variance(exp1_dqn_data, exp1_her_data, 'Experimento 1: Análisis de Varianza (DQN vs HER)', 'variance_exp1.png')
plot_and_save_variance(exp2_dqn_data, exp2_her_data, 'Experimento 2: Análisis de Varianza (DQN vs HER)', 'variance_exp2.png')
plot_and_save_variance(exp3_dqn_data, exp3_her_data, 'Experimento 3: Análisis de Varianza (DQN vs HER)', 'variance_exp3.png')