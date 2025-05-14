# run_ai_train.py

def main():
    print("Which AI would you like to train?")
    print("  1) Orbit Predictor (LSTM)")
    print("  2) Orbit Corrector (FNN)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        from training.predictor import train_predictor
        train_predictor()
    elif choice == "2":
        from training.corrector import train_corrector
        train_corrector()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()