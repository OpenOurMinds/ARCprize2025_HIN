import sys
import os

# Check if moved components can still be imported
def verify_imports():
    success = True
    try:
        # Check baselines
        if not os.path.exists('baselines'):
            print("❌ 'baselines' directory missing")
            success = False
        else:
            print("✅ 'baselines' directory exists")
            
        # Check data
        if not os.path.exists('data/json'):
            print("❌ 'data/json' directory missing")
            success = False
        else:
            print("✅ 'data/json' directory exists")
            
        # Check docs
        if not os.path.exists('docs'):
            print("❌ 'docs' directory missing")
            success = False
        else:
            print("✅ 'docs' directory exists")

    except Exception as e:
        print(f"❌ Verification failed with error: {e}")
        success = False
    
    if success:
        print("\n✨ Reorganization verified successfully!")
    else:
        print("\n⚠️  Reorganization verification found issues.")
    return success

if __name__ == "__main__":
    if verify_imports():
        sys.exit(0)
    else:
        sys.exit(1)
