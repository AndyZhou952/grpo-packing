# code generated with AI assitance
import os
import numpy as np
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

DEV_DIR = '/home/gawa/andy/mindrlhf_dev'
MASTER_DIR = '/home/gawa/zt_temp/mindrlhf_master'

PATTERNS = {
    'responses_mask': 'responses_mask_index_{index}_rank_{rank}.npy',
    'ref_per_token_logps': 'ref_per_token_logps_index_{index}_rank_{rank}.npy',
    'advantages': 'advantages_index_{index}_rank_{rank}.npy',
}

ATOL = 1e-3


def compare_files(dev_path, master_path, category, index, rank):
    """
    Compare two .npy files. Return True if mismatch found, False otherwise.
    Logs a warning with details if mismatch.
    """
    a = np.load(dev_path)
    b = np.load(master_path)

    shape_match = a.shape == b.shape
    max_a, max_b = a.max(), b.max()
    min_a, min_b = a.min(), b.min()
    close = np.allclose(a, b, atol=ATOL)

    if not (shape_match and np.isclose(max_a, max_b, atol=ATOL) \
            and np.isclose(min_a, min_b, atol=ATOL) and close):
        # Prepare mismatch details
        details = []
        if not shape_match:
            details.append(f"shape dev={a.shape}, master={b.shape}")
        if not np.isclose(max_a, max_b, atol=ATOL):
            details.append(f"max dev={max_a}, master={max_b}")
        if not np.isclose(min_a, min_b, atol=ATOL):
            details.append(f"min dev={min_a}, master={min_b}")
        if not close:
            diff = np.abs(a - b)
            details.append(f"allclose=False (max abs diff={diff.max()})")

        logging.warning(
            f"Mismatch [{category}] index={index}, rank={rank}: " + "; ".join(details)
        )
        return True

    return False


def main():
    setup_logging()

    # Counters: {category: {'total': int, 'mismatch': int}}
    stats = {cat: {'total': 0, 'mismatch': 0} for cat in PATTERNS}
    overall_total = 0
    overall_mismatch = 0

    for index in range(1, 65):
        for rank in range(8):
            for category, pattern in PATTERNS.items():
                fname = pattern.format(index=index, rank=rank)
                dev_path = os.path.join(DEV_DIR, fname)
                master_path = os.path.join(MASTER_DIR, fname)

                # Check existence
                missing = []
                if not os.path.isfile(dev_path):
                    missing.append(f"DEV missing: {dev_path}")
                if not os.path.isfile(master_path):
                    missing.append(f"MASTER missing: {master_path}")
                if missing:
                    for msg in missing:
                        logging.warning(msg)
                    continue

                # Both files exist: compare
                stats[category]['total'] += 1
                overall_total += 1

                mismatch = compare_files(dev_path, master_path, category, index, rank)
                if mismatch:
                    stats[category]['mismatch'] += 1
                    overall_mismatch += 1

    # Report percentages
    print("\n=== Comparison Summary ===")
    for category, vals in stats.items():
        total = vals['total']
        mism = vals['mismatch']
        pct = (mism / total * 100) if total > 0 else 0.0
        print(f"{category}: {mism}/{total} mismatches ({pct:.2f}%)")

    overall_pct = (overall_mismatch / overall_total * 100) if overall_total > 0 else 0.0
    print(f"Overall: {overall_mismatch}/{overall_total} mismatches ({overall_pct:.2f}%)")


if __name__ == '__main__':
    main()
