"""
Intelligent CSV Data Handler for CLV Platform
Handles ANY CSV format with smart column detection
Production-grade, defensive programming
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Optional, Tuple, List
import re
from datetime import datetime


class CLVDataHandler:
    """
    Intelligent handler for CLV data that automatically detects and maps columns.
    Handles messy real-world data with grace.
    """
    
    # Column detection patterns
    CUSTOMER_ID_PATTERNS = [
        'customer_id', 'customerid', 'customer', 'id', 'user_id', 'userid',
        'client_id', 'clientid', 'account_id', 'accountid'
    ]
    
    REVENUE_PATTERNS = [
        'revenue', 'sales', 'amount', 'value', 'purchase', 'spent', 'spend',
        'total', 'ltv', 'clv', 'transaction', 'order_value', 'payment'
    ]
    
    DATE_PATTERNS = [
        'date', 'timestamp', 'time', 'datetime', 'purchase_date', 'order_date',
        'transaction_date', 'created_at', 'created', 'occurred_at'
    ]
    
    FREQUENCY_PATTERNS = [
        'frequency', 'freq', 'count', 'purchases', 'orders', 'transactions',
        'visits', 'num_purchases', 'num_orders'
    ]
    
    RECENCY_PATTERNS = [
        'recency', 'days_since', 'last_purchase', 'days_ago', 'time_since'
    ]
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with raw dataframe"""
        self.raw_data = data.copy()
        self.data = None
        self.column_mapping = {}
        self.warnings = []
        self.errors = []
        
    def _normalize_column_name(self, col: str) -> str:
        """Normalize column name for matching"""
        return re.sub(r'[^a-z0-9]', '', col.lower().strip())
    
    def _detect_column(self, patterns: List[str], column_type: str) -> Optional[str]:
        """Detect column based on pattern matching"""
        normalized_cols = {self._normalize_column_name(col): col 
                          for col in self.raw_data.columns}
        
        # Try exact matches first
        for pattern in patterns:
            normalized_pattern = self._normalize_column_name(pattern)
            if normalized_pattern in normalized_cols:
                return normalized_cols[normalized_pattern]
        
        # Try partial matches
        for pattern in patterns:
            normalized_pattern = self._normalize_column_name(pattern)
            for norm_col, orig_col in normalized_cols.items():
                if normalized_pattern in norm_col:
                    return orig_col
        
        return None
    
    def _detect_customer_id(self) -> Optional[str]:
        """Detect customer ID column"""
        detected = self._detect_column(self.CUSTOMER_ID_PATTERNS, 'customer_id')
        
        if detected is None:
            # Look for column with unique values (likely an ID)
            for col in self.raw_data.columns:
                if self.raw_data[col].nunique() == len(self.raw_data):
                    self.warnings.append(f"Auto-detected customer ID: '{col}' (all unique values)")
                    return col
        
        return detected
    
    def _detect_revenue(self) -> Optional[str]:
        """Detect revenue column"""
        detected = self._detect_column(self.REVENUE_PATTERNS, 'revenue')
        
        if detected is None:
            # Look for numeric column with positive values
            for col in self.raw_data.columns:
                if pd.api.types.is_numeric_dtype(self.raw_data[col]):
                    if self.raw_data[col].min() >= 0:
                        self.warnings.append(f"Auto-detected revenue: '{col}' (positive numeric)")
                        return col
        
        return detected
    
    def _detect_date(self) -> Optional[str]:
        """Detect date column"""
        detected = self._detect_column(self.DATE_PATTERNS, 'date')
        
        if detected is None:
            # Try to parse columns as dates
            for col in self.raw_data.columns:
                try:
                    pd.to_datetime(self.raw_data[col], errors='coerce')
                    if pd.to_datetime(self.raw_data[col], errors='coerce').notna().sum() > len(self.raw_data) * 0.8:
                        self.warnings.append(f"Auto-detected date: '{col}' (parseable as datetime)")
                        return col
                except:
                    pass
        
        return detected
    
    def _aggregate_to_customer_level(self, customer_col: str, revenue_col: str, date_col: Optional[str]) -> pd.DataFrame:
        """
        Aggregate transaction-level data to customer-level.
        Calculates: total_revenue, frequency, recency, avg_order_value
        """
        df = self.raw_data.copy()
        
        # Convert revenue to numeric
        df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce').fillna(0)
        
        # Basic aggregation
        agg_dict = {
            revenue_col: ['sum', 'mean', 'count']
        }
        
        customer_df = df.groupby(customer_col).agg(agg_dict).reset_index()
        customer_df.columns = ['customer_id', 'total_revenue', 'avg_order_value', 'frequency']
        
        # Calculate recency if date available
        if date_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                max_date = df[date_col].max()
                
                last_purchase = df.groupby(customer_col)[date_col].max().reset_index()
                last_purchase['recency'] = (max_date - last_purchase[date_col]).dt.days
                
                customer_df = customer_df.merge(
                    last_purchase[['customer_id', 'recency']],
                    left_on='customer_id',
                    right_on='customer_id',
                    how='left'
                )
                customer_df['recency'] = customer_df['recency'].fillna(0)
                
                self.warnings.append("Calculated recency from transaction dates")
            except:
                customer_df['recency'] = 0
                self.warnings.append("Could not calculate recency - set to 0")
        else:
            customer_df['recency'] = 0
            self.warnings.append("No date column - recency set to 0")
        
        return customer_df
    
    def process(self) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Main processing function. Returns (success, processed_dataframe).
        """
        # Step 1: Detect required columns
        customer_col = self._detect_customer_id()
        revenue_col = self._detect_revenue()
        date_col = self._detect_date()
        
        # Step 2: Validate required columns
        if customer_col is None:
            self.errors.append(
                "❌ Could not detect CUSTOMER ID column. Expected: " + 
                ", ".join(self.CUSTOMER_ID_PATTERNS[:5]) + "..."
            )
            return False, None
        
        if revenue_col is None:
            self.errors.append(
                "❌ Could not detect REVENUE column. Expected: " + 
                ", ".join(self.REVENUE_PATTERNS[:5]) + "..."
            )
            return False, None
        
        # Step 3: Build column mapping
        self.column_mapping = {
            'customer_id': customer_col,
            'revenue': revenue_col,
            'date': date_col if date_col else 'NOT FOUND'
        }
        
        # Step 4: Check if data is transaction-level or customer-level
        n_customers = self.raw_data[customer_col].nunique()
        n_rows = len(self.raw_data)
        
        if n_rows > n_customers * 1.5:
            # Transaction-level data - need to aggregate
            self.warnings.append(f"Transaction-level data detected ({n_rows} rows, {n_customers} customers)")
            self.data = self._aggregate_to_customer_level(customer_col, revenue_col, date_col)
            self.warnings.append("Aggregated to customer-level")
        else:
            # Already customer-level
            self.data = self.raw_data.copy()
            self.data = self.data.rename(columns={
                customer_col: 'customer_id',
                revenue_col: 'total_revenue'
            })
            
            # Detect or create frequency
            freq_col = self._detect_column(self.FREQUENCY_PATTERNS, 'frequency')
            if freq_col:
                self.data['frequency'] = self.data[freq_col]
            else:
                self.data['frequency'] = 1
                self.warnings.append("No frequency column - set to 1")
            
            # Detect or create recency
            recency_col = self._detect_column(self.RECENCY_PATTERNS, 'recency')
            if recency_col:
                self.data['recency'] = self.data[recency_col]
            else:
                self.data['recency'] = 0
                self.warnings.append("No recency column - set to 0")
            
            # Calculate avg_order_value
            self.data['avg_order_value'] = self.data['total_revenue'] / self.data['frequency']
        
        # Step 5: Select final columns
        self.data = self.data[['customer_id', 'total_revenue', 'frequency', 'recency', 'avg_order_value']]
        
        # Step 6: Data quality checks
        if self.data['total_revenue'].isnull().any():
            self.errors.append("Some customers have null revenue values")
            return False, None
        
        if len(self.data) < 10:
            self.errors.append("Not enough customers for analysis (minimum 10)")
            return False, None
        
        return True, self.data
    
    def get_summary(self) -> Dict:
        """Get summary of processing"""
        return {
            'column_mapping': self.column_mapping,
            'warnings': self.warnings,
            'errors': self.errors,
            'total_rows': len(self.raw_data),
            'processed_customers': len(self.data) if self.data is not None else 0
        }
