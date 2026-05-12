"""Download the IBM Telco Customer Churn CSV into data/."""
from pathlib import Path
import urllib.request

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dest = root / "data" / "Telco-Customer-Churn.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already present: {dest}")
        return
    print(f"Downloading to {dest} ...")
    urllib.request.urlretrieve(URL, dest)
    print("Done.")


if __name__ == "__main__":
    main()
