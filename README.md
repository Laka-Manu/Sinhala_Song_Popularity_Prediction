Your README.md MUST include:
# Project Title

## Group Information
- **Student 1:** [Full Name as in LMS] - [Student ID] - Role: [Role Name]
- **Student 2:** [Full Name as in LMS] - [Student ID] - Role: [Role Name]
- **Student 3:** [Full Name as in LMS] - [Student ID] - Role: [Role Name]

## Project Description
[Brief description of what your application does]

## Live Deployment
🔗 **Live URL:** [Your deployed application URL]

## Technologies Used
- HTML5, CSS3, JavaScript
- [Any frameworks/libraries used]
- GitHub Actions
- [Deployment platform name]

## Features
- Feature 1
- Feature 2
- Feature 3

## Branch Strategy
We implemented the following branching strategy:
- `main` - Production branch
- `develop` - Integration branch
- `feature/*` - Feature development branches

## Individual Contributions

### [Student 1 Name]
- Repository setup and configuration
- GitHub Actions CI/CD pipeline implementation
- Deployment setup and management
- [List specific commits/features]

### [Student 2 Name]
- [List specific features developed]
- [List specific commits/contributions]

### [Student 3 Name]
- [List specific features developed]
- [List specific commits/contributions]

## Setup Instructions

### Prerequisites
- Node.js (version 18 or higher)
- Git

### Installation
```bash
# Clone the repository
git clone [your-repo-url]

# Navigate to project directory
cd [project-name]

# Install dependencies
npm install

# Run development server
npm run dev

# Deployment Process
[Explain how your CI/CD pipeline works]
# Challenges Faced
[Optional: Describe any challenges and how you resolved them]
# Build Status 
---

## 📝 Step-by-Step Implementation Guide

### Phase 1: Setup (0-15 minutes)

#### Step 1: Team Formation & Planning
1. Form your team (2-3 students)
2. Assign roles based on strengths
3. Choose your project type
4. Decide on features to implement

#### Step 2: Repository Creation (DevOps Engineer)
1. Go to GitHub.com and sign in
2. Click "New Repository"
3. Name: `[project-name]-devops-assignment`
4. Description: "Advanced Git & DevOps Assignment - [Group Number]"
5. Select **PUBLIC**
6. Initialize with README: **NO** (we'll create our own)
7. Click "Create Repository"

#### Step 3: Add Collaborators
1. Go to Settings → Collaborators
2. Add all team members by GitHub username
3. Each member should accept the invitation

#### Step 4: Clone Repository (All Members)
```bash
git clone https://github.com/[username]/[repo-name].git
cd [repo-name]

Step 5: Initial Setup (DevOps Engineer)
# Create initial files
touch README.md .gitignore

# Create branch structure
git checkout -b develop
git push -u origin develop

# Create GitHub Actions directory
mkdir -p .github/workflows

# Add initial commit
git add .
git commit -m "chore: initial repository setup"
git push origin develop
