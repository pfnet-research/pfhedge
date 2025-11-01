"""Script to replace print statements with logger calls in backtester.py"""

import re

# Read the file
with open('/Users/shaobaolin/Documents/repo/pfhedge/crypto/backtest/backtester.py', 'r') as f:
    content = f.read()

# Mapping of print patterns to logger replacements
replacements = [
    # Data loading
    (r'print\(f"Loading historical data from \{data_dir\}\.\.\."',
     'self.logger.log_data_loading_start(data_dir)'),

    (r'print\(f"Loading data from specified file: \{data_file\}"\)',
     'self.logger.log_data_loading_start(data_dir, data_file)'),

    (r'print\(f"✅ Loaded \{len\(perpetual_df\)\} raw records"\)',
     'self.logger.log_data_loaded(len(perpetual_df), "perpetual")'),

    (r'print\(f"Resampling to \{frequency\} frequency\.\.\."',
     'self.logger.log_resampling(frequency)'),

    (r'print\(\s*f"✅ Resampled to \{len\(resampled_df\)\} records at \{frequency\} intervals"\s*\)',
     'self.logger.log_resampled(len(resampled_df), frequency)'),

    (r'print\(\s*f"Filtering data from \{self\.config\.start_date\} to \{self\.config\.end_date\}\.\.\."',
     'self.logger.log_date_filtering(self.config.start_date, self.config.end_date)'),

    (r'print\(f"✅ Filtered to \{len\(filtered_df\)\} records in date range"\)',
     'self.logger.log_filtered(len(filtered_df))'),

    (r'print\(f"✅ Loaded \{len\(funding_in_range\)\} funding rate records in date range"\s*\)',
     'self.logger.log_funding_loaded(len(funding_in_range))'),

    (r'print\(f"⚠️  No funding data found \(this is okay, but funding costs won\'t be applied\)"\s*\)',
     'self.logger.log_no_funding_data()'),

    (r'print\(f"⚠️  No options data found \(this is okay for basic backtesting\)"\)',
     'self.logger.log_no_options_data()'),

    (r'print\(f"✅ Loaded \{len\(options_df\)\} options records in date range"\)',
     'self.logger.log_data_loaded(len(options_df), "options")'),
]

# Apply replacements
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# Write back
with open('/Users/shaobaolin/Documents/repo/pfhedge/crypto/backtest/backtester.py', 'w') as f:
    f.write(content)

print("Replacements complete!")
