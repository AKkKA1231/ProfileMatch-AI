import os
import json

# Ensure target directory exists
resumes_dir = os.path.join("data", "resumes")
os.makedirs(resumes_dir, exist_ok=True)

resumes_data = [
    # ML & AI Specialists
    {
        "filename": "jane_doe_sr_ml.txt",
        "format": "txt",
        "name": "Jane Doe",
        "title": "Senior Machine Learning Engineer",
        "exp": 6,
        "education": "M.S. in Computer Science, Stanford University",
        "skills": ["Python", "PyTorch", "Retrieval-Augmented Generation", "Transformers", "ChromaDB", "Kubernetes", "LangChain", "Docker"],
        "sections": {
            "Summary": "Senior Machine Learning Engineer with 6 years of experience building and deploying enterprise RAG pipelines, distributed LLM training, and high-throughput vector search engines.",
            "Work Experience": "Senior ML Engineer at TechCorp (2021 - Present):\n- Led deployment of enterprise RAG pipelines reducing query latency by 40% using ChromaDB and PyTorch.\n- Scaled distributed training workloads on Kubernetes clusters with 64 GPUs.\n- Designed section-aware chunking and automated metadata extraction algorithms for multi-modal document ingestion.\n\nMachine Learning Engineer at DataFlow Systems (2018 - 2021):\n- Built NLP classification models using Transformers and Hugging Face, achieving 94% F1-score.\n- Developed REST API endpoints using FastAPI for real-time inference.",
            "Education": "M.S. in Computer Science, Stanford University (2016 - 2018)\nB.S. in Computer Science, UC Berkeley (2012 - 2016)",
            "Skills": "Languages: Python, C++, SQL\nFrameworks: PyTorch, TensorFlow, Transformers, LangChain, LlamaIndex\nDatabases & Tools: ChromaDB, Pinecone, Kubernetes, Docker, Git",
            "Certifications": "AWS Certified Machine Learning - Specialty (2023)"
        }
    },
    {
        "filename": "alex_rivera_ml_researcher.txt",
        "format": "txt",
        "name": "Alex Rivera",
        "title": "AI Research Scientist & LLM Specialist",
        "exp": 8,
        "education": "Ph.D. in Artificial Intelligence, MIT",
        "skills": ["Python", "PyTorch", "LLMs", "DeepSpeed", "Distributed Training", "CUDA", "NLP", "BERT"],
        "sections": {
            "Summary": "AI Research Scientist specializing in Large Language Models (LLMs), deep learning architecture optimization, and distributed GPU training systems with 8 years of research and production experience.",
            "Work Experience": "Principal AI Scientist at Neural Labs (2020 - Present):\n- Co-authored architectures for 70B parameter LLM fine-tuning using DeepSpeed and PyTorch on 256 A100 GPUs.\n- Optimized CUDA kernels reducing memory overhead during attention computation by 25%.\n\nNLP Researcher at AI Institute (2016 - 2020):\n- Published 6 peer-reviewed papers in NeurIPS and ACL on semantic embeddings and transformer pre-training.",
            "Education": "Ph.D. in Artificial Intelligence, Massachusetts Institute of Technology (2012 - 2016)",
            "Skills": "Core: PyTorch, CUDA, DeepSpeed, Ray, Hugging Face, Python, C++",
            "Certifications": "NVIDIA Certified Deep Learning Professional"
        }
    },
    {
        "filename": "michael_chang_rag_dev.txt",
        "format": "txt",
        "name": "Michael Chang",
        "title": "Machine Learning Engineer - NLP & RAG",
        "exp": 4,
        "education": "B.S. in Computer Science, Carnegie Mellon University",
        "skills": ["Python", "LangChain", "LlamaIndex", "Vector Databases", "Weaviate", "FastAPI", "Docker", "PyTorch"],
        "sections": {
            "Summary": "Machine Learning Engineer with 4 years of experience building RAG systems, semantic search integrations, and automated metadata extraction pipelines.",
            "Work Experience": "ML Engineer at SearchAI (2020 - Present):\n- Integrated Weaviate vector database with LangChain to power semantic search for 500k documents.\n- Implemented hybrid search combining BM25 keyword matching with dense vector retrieval.\n- Created automated document ingestion pipelines using FastAPI and PyMuPDF.",
            "Education": "B.S. in Computer Science, Carnegie Mellon University (2016 - 2020)",
            "Skills": "Python, PyTorch, LangChain, Weaviate, Pinecone, Docker, REST APIs"
        }
    },
    {
        "filename": "sarah_jenkins_junior_ml.txt",
        "format": "txt",
        "name": "Sarah Jenkins",
        "title": "Junior ML Engineer",
        "exp": 2,
        "education": "B.S. in Data Science, University of Washington",
        "skills": ["Python", "scikit-learn", "Pandas", "TensorFlow", "SQL", "Git"],
        "sections": {
            "Summary": "Enthusiastic Junior ML Engineer with 2 years of experience in data preprocessing, feature engineering, and predictive modeling.",
            "Work Experience": "Junior Data Scientist at Analytics Co (2022 - Present):\n- Built customer churn prediction models using scikit-learn and XGBoost.\n- Automated ETL pipelines using Python and SQL.",
            "Education": "B.S. in Data Science, University of Washington (2018 - 2022)",
            "Skills": "Python, SQL, Pandas, NumPy, scikit-learn, Git"
        }
    },
    
    # Frontend Engineers
    {
        "filename": "david_kim_staff_frontend.txt",
        "format": "txt",
        "name": "David Kim",
        "title": "Staff Frontend Engineer",
        "exp": 7,
        "education": "B.S. in Computer Science, UCLA",
        "skills": ["React", "TypeScript", "Next.js", "Redux", "Tailwind CSS", "Jest", "GraphQL", "Web Performance"],
        "sections": {
            "Summary": "Staff Frontend Engineer with 7 years of expertise in React, TypeScript, Next.js, and enterprise design systems. Proven track record scaling web applications to millions of active users.",
            "Work Experience": "Staff Frontend Engineer at WebScale Inc (2021 - Present):\n- Architected core Next.js application reducing initial page load time by 50%.\n- Built enterprise component library adopted across 12 product teams.\n\nSenior Frontend Developer at UI Solutions (2017 - 2021):\n- Built complex state management solutions using React, Redux Toolkit, and GraphQL.",
            "Education": "B.S. in Computer Science, UCLA (2013 - 2017)",
            "Skills": "TypeScript, React, Next.js, Tailwind CSS, Jest, Playwright, Webpack"
        }
    },
    {
        "filename": "emily_watson_sr_frontend.txt",
        "format": "txt",
        "name": "Emily Watson",
        "title": "Senior Frontend Developer",
        "exp": 5,
        "education": "B.A. in Interactive Media, NYU",
        "skills": ["JavaScript", "TypeScript", "React", "HTML5", "CSS3", "Zustand", "REST APIs", "Cypress"],
        "sections": {
            "Summary": "Senior Frontend Developer with 5 years experience specializing in responsive web design, React state management, and modern CSS animations.",
            "Work Experience": "Senior Frontend Developer at CreativeTech (2020 - Present):\n- Developed responsive web interfaces using React, TypeScript, and CSS Modules.\n- Implemented end-to-end component testing using Cypress.",
            "Education": "B.A. in Interactive Media, NYU (2015 - 2019)",
            "Skills": "React, JavaScript, TypeScript, HTML5, CSS3, Cypress, Figma"
        }
    },
    {
        "filename": "marcus_johnson_react_dev.txt",
        "format": "txt",
        "name": "Marcus Johnson",
        "title": "Frontend Engineer - React Specialist",
        "exp": 3,
        "education": "B.S. in Software Engineering, UT Austin",
        "skills": ["React", "JavaScript", "Redux", "CSS3", "Git", "Bootstrap"],
        "sections": {
            "Summary": "Frontend Engineer with 3 years experience building dynamic web apps with React and Redux.",
            "Work Experience": "Frontend Engineer at AppStudio (2021 - Present):\n- Built interactive dashboards using React, Redux, and Chart.js.",
            "Education": "B.S. in Software Engineering, UT Austin (2017 - 2021)",
            "Skills": "React, JavaScript, HTML/CSS, Redux, Git"
        }
    },
    
    # DevOps & SRE Specialists
    {
        "filename": "robert_chen_sr_devops.txt",
        "format": "txt",
        "name": "Robert Chen",
        "title": "Senior DevOps & Infrastructure Engineer",
        "exp": 6,
        "education": "B.S. in Computer Engineering, UIUC",
        "skills": ["Kubernetes", "Terraform", "AWS", "Docker", "Ansible", "Prometheus", "Grafana", "Python", "CI/CD"],
        "sections": {
            "Summary": "Senior DevOps & SRE Engineer with 6 years experience managing cloud infrastructure on AWS, building IaC with Terraform, and orchestrating Kubernetes clusters.",
            "Work Experience": "Lead Infrastructure Engineer at CloudOps (2020 - Present):\n- Designed multi-region AWS infrastructure using Terraform and Ansible.\n- Managed 50+ EKS Kubernetes clusters with Istio service mesh.\n- Built automated CI/CD pipelines using GitHub Actions.",
            "Education": "B.S. in Computer Engineering, UIUC (2014 - 2018)",
            "Skills": "AWS, Kubernetes, Docker, Terraform, Prometheus, Python, Bash",
            "Certifications": "AWS Certified Solutions Architect - Professional, CKA (Certified Kubernetes Administrator)"
        }
    },
    {
        "filename": "priya_sharma_sre.txt",
        "format": "txt",
        "name": "Priya Sharma",
        "title": "Site Reliability Engineer (SRE)",
        "exp": 4,
        "education": "B.Tech in Computer Science, IIT Delhi",
        "skills": ["Linux", "Python", "Go", "Docker", "Kubernetes", "Datadog", "Terraform", "GCP"],
        "sections": {
            "Summary": "Site Reliability Engineer with 4 years experience optimizing system reliability, incident management, and automated monitoring on Google Cloud Platform.",
            "Work Experience": "SRE at FinTech Global (2020 - Present):\n- Maintained 99.99% uptime for payment gateway microservices on GCP.\n- Created custom SLO/SLI dashboards in Datadog and Grafana.",
            "Education": "B.Tech in Computer Science, IIT Delhi (2016 - 2020)",
            "Skills": "Go, Python, Linux, GCP, Kubernetes, Datadog, Terraform"
        }
    },
    
    # Data Engineers
    {
        "filename": "carlos_mendoza_data_lead.txt",
        "format": "txt",
        "name": "Carlos Mendoza",
        "title": "Data Engineering Lead",
        "exp": 8,
        "education": "M.S. in Information Systems, Northwestern University",
        "skills": ["Apache Spark", "Python", "SQL", "Snowflake", "Kafka", "Airflow", "dbt", "Databricks", "AWS"],
        "sections": {
            "Summary": "Data Engineering Lead with 8 years experience architecting large-scale batch and real-time data pipelines using Spark, Kafka, Snowflake, and Airflow.",
            "Work Experience": "Data Engineering Lead at Enterprise Analytics (2019 - Present):\n- Architected cloud data warehouse on Snowflake processing 10TB daily data ingest.\n- Built real-time streaming data pipelines using Apache Kafka and Spark Streaming.\n- Managed Airflow DAG orchestrations for 200+ data workflows.",
            "Education": "M.S. in Information Systems, Northwestern University (2014 - 2016)",
            "Skills": "Apache Spark, Kafka, Airflow, Snowflake, Python, SQL, dbt, AWS"
        }
    },
    {
        "filename": "amanda_foster_sr_data_eng.txt",
        "format": "txt",
        "name": "Amanda Foster",
        "title": "Senior Data Engineer",
        "exp": 5,
        "education": "B.S. in Applied Mathematics, University of Michigan",
        "skills": ["Python", "SQL", "BigQuery", "Apache Spark", "Airflow", "GCP", "PostgreSQL"],
        "sections": {
            "Summary": "Senior Data Engineer with 5 years experience building reliable ETL pipelines and data modeling on GCP and BigQuery.",
            "Work Experience": "Senior Data Engineer at DataMetrics (2020 - Present):\n- Built scalable ETL pipelines in Python and SQL using Google BigQuery and Airflow.\n- Designed dimensional data models for business intelligence reporting.",
            "Education": "B.S. in Applied Mathematics, University of Michigan (2015 - 2019)",
            "Skills": "Python, SQL, BigQuery, Airflow, GCP, PostgreSQL, Git"
        }
    },
    
    # Product Managers
    {
        "filename": "jessica_taylor_sr_pm.txt",
        "format": "txt",
        "name": "Jessica Taylor",
        "title": "Senior Technical Product Manager - AI Products",
        "exp": 6,
        "education": "MBA, Harvard Business School | B.S. in Computer Science, MIT",
        "skills": ["Product Management", "AI/ML Strategy", "Agile/Scrum", "User Research", "Data Analytics", "SQL", "Roadmapping"],
        "sections": {
            "Summary": "Senior Technical Product Manager with 6 years experience launching AI-driven B2B SaaS applications, leading cross-functional engineering teams, and driving product roadmaps.",
            "Work Experience": "Senior Technical Product Manager at AI SaaS Inc (2020 - Present):\n- Spearheaded launch of enterprise generative AI assistant generating $5M ARR in Year 1.\n- Managed product roadmap, user stories, and feature prioritization across 3 engineering squads.",
            "Education": "MBA, Harvard Business School (2018 - 2020)\nB.S. in Computer Science, MIT (2012 - 2016)",
            "Skills": "Product Strategy, User Research, Agile, JIRA, SQL, Mixpanel, Roadmapping"
        }
    },
    {
        "filename": "kevin_patel_pm.txt",
        "format": "txt",
        "name": "Kevin Patel",
        "title": "Product Manager",
        "exp": 4,
        "education": "B.S. in Business & Computer Science, University of Toronto",
        "skills": ["Product Management", "Wireframing", "A/B Testing", "Agile", "SQL", "Mixpanel"],
        "sections": {
            "Summary": "Product Manager with 4 years experience driving user growth and feature delivery for mobile and web SaaS applications.",
            "Work Experience": "Product Manager at TechGrowth (2020 - Present):\n- Led customer discovery and wireframing for onboarding flow, increasing conversion by 22%.",
            "Education": "B.S. in Business & CS, University of Toronto (2016 - 2020)",
            "Skills": "Agile, JIRA, Wireframing, SQL, Mixpanel"
        }
    }
]

# Additional candidates to reach 30 diverse profiles across formats (.txt, .docx, .pdf simulated)
extra_candidates = [
    ("Brian Vance", "Full Stack Developer", 4, "React, Node.js, TypeScript, PostgreSQL, Docker", "brian_vance_fullstack.txt"),
    ("Rachel Green", "QA Automation Engineer", 5, "Python, Selenium, PyTest, Cypress, CI/CD, Jenkins", "rachel_green_qa.txt"),
    ("Daniel White", "Cloud Security Engineer", 6, "AWS, IAM, Terraform, Python, Kubernetes Security, SIEM", "daniel_white_sec.txt"),
    ("Sophia Martinez", "UI/UX Developer", 3, "HTML5, CSS3, JavaScript, React, Figma, Tailwind", "sophia_martinez_ui.txt"),
    ("James Wilson", "Backend Python Engineer", 5, "Python, Django, FastAPI, PostgreSQL, Redis, Celery", "james_wilson_backend.txt"),
    ("Laura Bennett", "Machine Learning Engineer", 3, "Python, PyTorch, Hugging Face, NLP, Scikit-learn", "laura_bennett_ml.txt"),
    ("Vikram Singh", "Distributed Systems Engineer", 7, "C++, Go, Kubernetes, gRPC, Distributed Storage", "vikram_singh_systems.txt"),
    ("Hannah Abbott", "Data Scientist", 4, "Python, R, SQL, Machine Learning, Tableau, Pandas", "hannah_abbott_ds.txt"),
    ("Nathan Drake", "Mobile Engineer - React Native", 5, "React Native, TypeScript, iOS, Android, Redux", "nathan_drake_mobile.txt"),
    ("Olivia Parker", "DevOps Engineer", 3, "Docker, Kubernetes, AWS, Terraform, GitHub Actions", "olivia_parker_devops.txt"),
    ("Ethan Hunt", "Cybersecurity Analyst", 5, "Network Security, Penetration Testing, Python, Linux", "ethan_hunt_cyber.txt"),
    ("Chloe Adams", "Frontend Developer", 2, "JavaScript, HTML, CSS, React, Git", "chloe_adams_fe.txt"),
    ("Gabriel Silva", "MLOps Engineer", 5, "Python, MLflow, Kubeflow, PyTorch, Docker, AWS", "gabriel_silva_mlops.txt"),
    ("Zoe Collins", "Database Administrator", 6, "PostgreSQL, MySQL, Oracle, Performance Tuning, SQL", "zoe_collins_dba.txt"),
    ("Lucas Scott", "Software Engineer - C++", 4, "C++, STL, Multi-threading, Linux, CMake, Data Structures", "lucas_scott_cpp.txt"),
    ("Maya Lin", "Data Engineer", 3, "Python, SQL, Spark, Airflow, PostgreSQL", "maya_lin_data.txt"),
    ("Liam Neeson", "Site Reliability Engineer", 7, "Linux, Go, Kubernetes, Prometheus, Incident Response", "liam_neeson_sre.txt")
]

for name, title, exp, skills, filename in extra_candidates:
    resumes_data.append({
        "filename": filename,
        "format": "txt",
        "name": name,
        "title": title,
        "exp": exp,
        "education": "B.S. in Computer Science",
        "skills": [s.strip() for s in skills.split(",")],
        "sections": {
            "Summary": f"{title} with {exp} years of experience in software development and technology solutions.",
            "Work Experience": f"{title} at Tech Solutions (2020 - Present):\n- Delivered key features and engineering projects using {skills}.\n- Collaborated with cross-functional teams to optimize performance.",
            "Education": "B.S. in Computer Science, State University (2016 - 2020)",
            "Skills": f"Core Competencies: {skills}"
        }
    })

# Write resumes to data/resumes directory
count = 0
for r in resumes_data:
    filepath = os.path.join(resumes_dir, r["filename"])
    content = f"Name: {r['name']}\nTitle: {r['title']}\nTotal Experience: {r['exp']} years\n\n"
    for sec_name, sec_body in r["sections"].items():
        content += f"=== {sec_name.upper()} ===\n{sec_body}\n\n"
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    count += 1

print(f"Successfully generated {count} resumes in {resumes_dir}")
