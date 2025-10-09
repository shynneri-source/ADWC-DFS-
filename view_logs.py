#!/usr/bin/env python
"""
Script to view and manage training logs
"""

import os
import argparse
from datetime import datetime


def list_logs(log_dir='logs', prefix=None):
    """List all log files"""
    if not os.path.exists(log_dir):
        print(f"Log directory '{log_dir}' does not exist")
        return []
    
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    
    if prefix:
        log_files = [f for f in log_files if f.startswith(prefix)]
    
    if not log_files:
        print(f"No log files found in '{log_dir}'")
        return []
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda f: os.path.getmtime(os.path.join(log_dir, f)), reverse=True)
    
    print(f"\n{'#'*80}")
    print(f"{'Log Files':^80}")
    print(f"{'#'*80}\n")
    print(f"{'#':<4} {'Filename':<40} {'Size':<12} {'Modified':<20}")
    print(f"{'-'*80}")
    
    for idx, filename in enumerate(log_files, 1):
        filepath = os.path.join(log_dir, filename)
        size = os.path.getsize(filepath)
        size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
        mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
        mtime_str = mtime.strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"{idx:<4} {filename:<40} {size_str:<12} {mtime_str:<20}")
    
    print(f"{'-'*80}\n")
    
    return [os.path.join(log_dir, f) for f in log_files]


def view_log(log_file, lines=None, tail=False):
    """View content of a log file"""
    if not os.path.exists(log_file):
        print(f"Log file '{log_file}' does not exist")
        return
    
    print(f"\n{'='*80}")
    print(f"Log File: {log_file}")
    print(f"{'='*80}\n")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.readlines()
    
    if lines:
        if tail:
            content = content[-lines:]
        else:
            content = content[:lines]
    
    for line in content:
        print(line, end='')
    
    print(f"\n\n{'='*80}")
    print(f"End of log file")
    print(f"{'='*80}\n")


def search_logs(log_dir='logs', keyword='', prefix=None):
    """Search for keyword in log files"""
    if not os.path.exists(log_dir):
        print(f"Log directory '{log_dir}' does not exist")
        return
    
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    
    if prefix:
        log_files = [f for f in log_files if f.startswith(prefix)]
    
    if not log_files:
        print(f"No log files found")
        return
    
    print(f"\n{'='*80}")
    print(f"Searching for '{keyword}' in {len(log_files)} log file(s)")
    print(f"{'='*80}\n")
    
    for log_file in log_files:
        filepath = os.path.join(log_dir, log_file)
        matches = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if keyword.lower() in line.lower():
                    matches.append((line_num, line.rstrip()))
        
        if matches:
            print(f"\n📄 {log_file} ({len(matches)} matches)")
            print(f"{'-'*80}")
            for line_num, line in matches[:10]:  # Show first 10 matches
                print(f"  Line {line_num}: {line}")
            
            if len(matches) > 10:
                print(f"  ... and {len(matches) - 10} more matches")


def main():
    parser = argparse.ArgumentParser(description='View and manage ADWC-DFS training logs')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory containing log files')
    parser.add_argument('--list', action='store_true',
                       help='List all log files')
    parser.add_argument('--view', type=int, default=None,
                       help='View log file by number (from list)')
    parser.add_argument('--latest', action='store_true',
                       help='View the latest log file')
    parser.add_argument('--file', type=str, default=None,
                       help='View specific log file')
    parser.add_argument('--lines', type=int, default=None,
                       help='Number of lines to show')
    parser.add_argument('--tail', action='store_true',
                       help='Show last N lines (use with --lines)')
    parser.add_argument('--prefix', type=str, default=None,
                       help='Filter logs by prefix (training, evaluation, demo)')
    parser.add_argument('--search', type=str, default=None,
                       help='Search for keyword in log files')
    
    args = parser.parse_args()
    
    if args.search:
        search_logs(args.log_dir, args.search, args.prefix)
    elif args.list or (not args.view and not args.latest and not args.file):
        list_logs(args.log_dir, args.prefix)
    elif args.latest:
        log_files = list_logs(args.log_dir, args.prefix)
        if log_files:
            view_log(log_files[0], args.lines, args.tail)
    elif args.view:
        log_files = list_logs(args.log_dir, args.prefix)
        if log_files and 1 <= args.view <= len(log_files):
            view_log(log_files[args.view - 1], args.lines, args.tail)
        else:
            print(f"Invalid log number: {args.view}")
    elif args.file:
        view_log(args.file, args.lines, args.tail)


if __name__ == '__main__':
    main()
