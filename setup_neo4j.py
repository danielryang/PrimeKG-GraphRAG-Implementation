#!/usr/bin/env python3
"""
Script to set up Neo4j for PrimeKG-GraphRAG system.
Provides multiple deployment options.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_docker():
    """Check if Docker is available."""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def setup_neo4j_docker():
    """Set up Neo4j using Docker."""
    print("Setting up Neo4j using Docker...")
    
    # Check if Docker is available
    if not check_docker():
        print("ERROR: Docker is not available or not running")
        print("Please install Docker Desktop or start Docker service")
        return False
    
    # Create data directory
    data_dir = Path('./neo4j_data')
    data_dir.mkdir(exist_ok=True)
    
    # Stop any existing Neo4j container
    print("Stopping any existing Neo4j container...")
    subprocess.run(['docker', 'stop', 'primekg-neo4j'], capture_output=True)
    subprocess.run(['docker', 'rm', 'primekg-neo4j'], capture_output=True)
    
    # Start Neo4j container
    docker_cmd = [
        'docker', 'run', '-d',
        '--name', 'primekg-neo4j',
        '-p', '7474:7474',  # HTTP
        '-p', '7687:7687',  # Bolt
        '-v', f'{data_dir.absolute()}:/data',
        '-e', 'NEO4J_AUTH=neo4j/primekg123',
        '--restart=unless-stopped',
        'neo4j:latest'
    ]
    
    print("Starting Neo4j container...")
    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("SUCCESS: Neo4j container started successfully")
            print("Neo4j will be available at:")
            print("  - Web interface: http://localhost:7474")
            print("  - Bolt connection: bolt://localhost:7687")
            print("  - Username: neo4j")
            print("  - Password: primekg123")
            print("\nWaiting for Neo4j to start up (this may take 30-60 seconds)...")
            return True
        else:
            print(f"ERROR: Failed to start Neo4j container")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("ERROR: Docker command timed out")
        return False

def create_env_config():
    """Create .env configuration for Neo4j."""
    env_file = Path('.env')
    
    neo4j_config = """
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=primekg123
NEO4J_DATABASE=neo4j

# Memory Configuration (optimized)
MAX_MEMORY_USAGE=4.0
GRAPHRAG_CACHE_SEARCH=512
GRAPHRAG_CACHE_NEIGHBORS=256
GRAPHRAG_CACHE_PATHS=128
GRAPHRAG_CACHE_ENCODINGS=128

# Performance Settings
GRAPHRAG_USE_CACHE=true
PRIMEKG_AUTO_DOWNLOAD=true
USE_SPACY=true
SIMILARITY_THRESHOLD=0.3
QUERY_TIMEOUT=30
"""
    
    # Read existing .env or create new
    existing_env = ""
    if env_file.exists():
        existing_env = env_file.read_text()
    
    # Add Neo4j config if not present
    if "NEO4J_URI" not in existing_env:
        with open(env_file, 'a') as f:
            f.write(neo4j_config)
        print(f"Updated {env_file} with Neo4j configuration")
    else:
        print(f"Neo4j configuration already exists in {env_file}")
    
    return True

def test_neo4j_connection():
    """Test Neo4j connection."""
    print("\nTesting Neo4j connection...")
    
    try:
        # Add project root to path
        project_root = Path('.').absolute()
        sys.path.insert(0, str(project_root))
        
        from src.graph_data_source import PrimeKGDataSource
        
        # Create data source (will try Neo4j connection)
        ds = PrimeKGDataSource()
        
        if ds.using_neo4j:
            print("SUCCESS: Neo4j connection working!")
            return True
        else:
            print("WARNING: Could not connect to Neo4j, will use PyKEEN fallback")
            return False
            
    except Exception as e:
        print(f"ERROR: Neo4j connection test failed: {e}")
        return False

def setup_cloud_instructions():
    """Provide instructions for Neo4j cloud setup."""
    print("\n" + "="*60)
    print("NEO4J AURA CLOUD SETUP INSTRUCTIONS")
    print("="*60)
    print("""
For production deployment, consider Neo4j Aura (cloud):

1. Go to https://neo4j.com/aura/
2. Create a free account and new database
3. Note your connection details:
   - URI: neo4j+s://xxxxx.databases.neo4j.io
   - Username: neo4j  
   - Password: (your generated password)

4. Update your .env file:
   NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-password
   NEO4J_DATABASE=neo4j

5. Run the system - it will automatically use cloud Neo4j

Benefits of Neo4j Aura:
- Fully managed (no Docker required)
- Automatic backups and scaling
- Production-ready security
- 10-50x performance improvement over PyKEEN
""")

def main():
    """Main setup function."""
    print("PrimeKG-GraphRAG Neo4j Setup")
    print("="*40)
    
    print("\nChoose Neo4j setup option:")
    print("1. Docker (local development)")
    print("2. Cloud setup instructions")
    print("3. Just create .env config")
    print("4. Test existing connection")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        success = setup_neo4j_docker()
        if success:
            create_env_config()
            # Wait a bit for Neo4j to start
            import time
            print("Waiting 30 seconds for Neo4j to fully start...")
            time.sleep(30)
            test_neo4j_connection()
        
    elif choice == "2":
        setup_cloud_instructions()
        create_env_config()
        
    elif choice == "3":
        create_env_config()
        print("Configuration created. Set up Neo4j separately.")
        
    elif choice == "4":
        test_neo4j_connection()
        
    else:
        print("Invalid choice")
        return False
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print("Next steps:")
    print("1. Verify Neo4j is running (check http://localhost:7474)")
    print("2. Run notebooks to test the system")
    print("3. If Neo4j fails, system will automatically fall back to PyKEEN")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)