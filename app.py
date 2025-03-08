import argparse
import torch
import torchaudio
import os

from resemble_enhance.enhancer.inference import denoise, enhance

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def process_audio(input_path, output_path, solver, nfe, tau, denoising):
    if not os.path.isfile(input_path):
        print(f"Error: File '{input_path}' not found.")
        return
    
    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = torchaudio.load(input_path)
    dwav = dwav.mean(dim=0)

    if denoising:
        dwav, sr = denoise(dwav, sr, device)
    
    enhanced_wav, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)
    
    torchaudio.save(output_path, enhanced_wav.unsqueeze(0).cpu(), new_sr)
    print(f"Enhanced audio saved at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="CLI tool for AI-driven audio enhancement.")
    parser.add_argument("input", type=str, help="Path to the input audio file")
    parser.add_argument("output", type=str, help="Path to save the enhanced audio file")
    parser.add_argument("--solver", type=str, choices=["midpoint", "rk4", "euler"], default="midpoint", help="CFM ODE Solver (default: Midpoint)")
    parser.add_argument("--nfe", type=int, default=64, help="CFM Number of Function Evaluations (default: 64)")
    parser.add_argument("--tau", type=float, default=0.5, help="CFM Prior Temperature (default: 0.5)")
    parser.add_argument("--denoise", action="store_true", help="Apply denoising before enhancement")
    
    args = parser.parse_args()
    
    process_audio(args.input, args.output, args.solver, args.nfe, args.tau, args.denoise)

if __name__ == "__main__":
    main()
