"""Paper trade ledger persisting and tracking order records matching canonical schema."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from prooflab.backtest.orders import OrderRecord


class PaperTradeLedger:
    """Immutable audit ledger recording all simulated paper trade executions."""

    def __init__(self, storage_path: Path | str | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self._trades: list[OrderRecord] = []

        if self.storage_path and self.storage_path.exists():
            self.load_from_disk()

    @property
    def trades(self) -> list[OrderRecord]:
        return list(self._trades)

    def record_trade(self, trade: OrderRecord) -> None:
        """Append a closed or updated trade to the ledger."""
        self._trades.append(trade)
        if self.storage_path:
            self.save_to_disk()

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Convert recorded trades into a standardized pandas DataFrame."""
        if not self._trades:
            return pd.DataFrame()
        records = [t.model_dump(mode="json") for t in self._trades]
        df = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "fill_timestamp" in df.columns:
            df["fill_timestamp"] = pd.to_datetime(df["fill_timestamp"])
        if "exit_timestamp" in df.columns:
            df["exit_timestamp"] = pd.to_datetime(df["exit_timestamp"])
        return df

    def save_to_disk(self, file_path: Path | str | None = None) -> None:
        """Persist recorded trades as JSON."""
        target = Path(file_path) if file_path else self.storage_path
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            data = [t.model_dump(mode="json") for t in self._trades]
            target.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_from_disk(self, file_path: Path | str | None = None) -> None:
        """Load trade history from JSON file."""
        target = Path(file_path) if file_path else self.storage_path
        if target and target.exists():
            content = target.read_text(encoding="utf-8")
            if content.strip():
                items = json.loads(content)
                self._trades = [OrderRecord.model_validate(item) for item in items]
