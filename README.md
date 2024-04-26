### Eni6ma.org : The only password you will never use!

#### By Dylan Rosario @soltrinox (Inventor of Rosario Cypher)
#### Copyright 2024 All Rights Reserved

---

 - [Cybersecurity ](CyberSecurity.Overview.md) 
 - [Quantum cryptanalysis](Quantum.Cryptanalysis.md) 
 - [Cost of quantum computer](Cost.QuantumComputers.md)
 - [Secure by design](Secure.By.Design.md)
 - [Security patterns](Security.Patterns.md)
 - [Cryptographic primitives](Cryptographic.Primitives.md)

  ---

## Security Patterns

1. **[Authentication](Security.Patterns.md#1-authentication)**: Shows the steps from user input through to system verification of credentials.
2. **[Verification](Security.Patterns.md#2-verification)**: Details the process from data submission, hash generation, digital signature creation, to verification and outcome.
3. **[Access Control](Security.Patterns.md#3-access-control)**: Follows the login to the access request, role verification, and access decision.
4. **[Permissions](Security.Patterns.md#4-permissions)**: Covers the setting up of permissions by an admin to the checking of permissions during a user operation.
5. **[Authority and Rights](Security.Patterns.md#5-authority-and-rights)**: Details the assignment of rights and checks performed during a user action.
6. **[Voting Ballot](Security.Patterns.md#6-voting-ballot)**: Follows the process of secure voting from login to encrypted submission.
7. **[Certification](Security.Patterns.md#7-certification)**: Covers the entire lifecycle of a digital certificate from request to usage and verification.
8. **[Private Identity](Security.Patterns.md#8-private-identity)**: Describes steps involved in creating and managing a private identity securely.
9. **[Sovereign Identity](Security.Patterns.md#9-sovereign-identity)**: Outlines the creation and management of a sovereign identity using blockchain technology.
10. **[Single Sign-On (SSO)](Security.Patterns.md#10-single-sign-on-sso)**: Shows the process from initial login through to token verification and access outcome.
11. **[Consensus](Security.Patterns.md#11-consensus)**: Details the steps from transaction initiation to the consensus process and final blockchain update.
12. **[Authority](Security.Patterns.md#12-authority)**: Follows the role assignment to the authentication, authorization, and access decision processes, including audit logging.
13. **[Provenance](Security.Patterns.md#13-provenance)**: Covers everything from data creation, processing, and review to audits and decision-making based on verified data.
14. **[Non-Repudiation](Security.Patterns.md#14-non-repudiation)**: Describes the sequence from message creation and digital signing to signature verification and validation of non-denial.
15. **[Validation of Ownership](Security.Patterns.md#15-validation-of-ownership)**: Shows the steps from ownership claim through credential verification to the validation of ownership based on matching credentials and rights.
16. **[Verification of Ledger Log](Security.Patterns.md#16-verification-of-data)**: Details the process from transaction initiation, through authentication and signing, to the consensus process and final ledger update.
17. **[Irrefutable Evidence](Security.Patterns.md#17-irrefutable-evidence)**: Covers the sequence from data capture, cryptographic sealing, secure storage, to the availability of verification tools and the use of data in legal and compliance contexts.




## Cryptographic Primitives

 - Coprime (Relatively Prime)
 - XOR (Exclusive OR):
 - Modulus Operation
 - Math Notation Key
 - Generic Cryptography Algorithm
 - Shift
 - Rotate
 - Logarithm
 - Encrypt & Decrypt
 - Example Rust Implementation
 [READ MORE](Cryptographic.Primitives.md)


## Cyber Security Overview
In today's digital world, security is paramount.  This is especially true when it comes to verifying the identities of users and the authenticity of data. This paper explores the critical concepts of authentication and verification, untangling their differences and highlighting their roles in cybersecurity. We will delve into the various methods employed to achieve these goals, examining how users prove who they are and how data integrity is ensured. From widely used passwords to cutting-edge biometrics, we'll explore the factors that underpin authentication protocols. We'll also dissect the secure storage and management of user secrets, a cornerstone of robust authentication systems. By understanding these concepts, we can navigate the digital landscape with greater confidence and security.   [READ MORE](CyberSecurity.Overview.md) 


## Quantum Cryptanalysis

Despite the remarkable theoretical potential of quantum computing for cryptanalysis, its practical application remains largely theoretical and confined to experimental physics. The current technological state does not support the sensationalized claims of imminent threats to existing cryptographic systems. It is crucial for both the scientific community and the public to maintain a balanced perspective on the capabilities and limitations of quantum computers, steering clear of the pitfalls of hype and misinformation. Looking forward, continued research and development are vital to bridge the gap between the theoretical potential and practical application of quantum cryptanalysis, ensuring preparedness for future advancements in quantum computing.  [READ MORE](Quantum.Cryptanalysis.md) 

## The Cost of Quantum Computing

Quantum computing emerges as a pioneering force in the landscape of technological innovation, poised to revolutionize a spectrum of industries spanning from healthcare to finance. Despite the allure of sensational headlines underscoring their potential, the journey toward realizing the full capabilities of quantum computers is fraught with formidable challenges. This comprehensive exploration endeavors to shed light on the formidable investments requisite for both the development and operation of quantum computing systems.  [READ MORE](Cost.QuantumComputers.md)


## Secure By Design

The Rosario Proof system and Cypher are new cryptographic tools designed to work without relying on electronic devices. This makes them suitable for various environments, including mobile devices, computers, and even isolated systems with no internet connection. The system is built with "Secure by Design" in mind, where security features are integrated from the beginning. The document argues that strong security requires a layered approach. It emphasizes the importance of building security on a foundation of cryptographic primitives like hash functions and digital signatures. These primitives are then used to create more complex security patterns such as access control and digital identities. Finally, the paper explores various security concepts in detail. These include how systems determine who has access (authority), how data history is tracked (provenance), and how to ensure actions cannot be denied (non-repudiation). It also explains how common security patterns like digital signatures and blockchain technology are used to implement these concepts. [READ MORE](Secure.By.Design.md)



---
## Quantum Attacks on Contemporary Cyphers

The best method known for solving languages in NP deterministically uses exponential time. (Remember, computers are deterministic by their very nature.) NP $i$ (exponential time) defined as :

$$
\bigcup_{k}=\tau\left(2 n^{k}\right)
$$

Exponential time algorithms typically arise when we solve problems by exhaustively searching through a space of solutions, called brute-force search. Ideally the language of any cryptographic scheme forces NP $i$ computation upon any Turing machine.

### Table 1: Complexity and Key Space Analysis for Grover's Algorithm

| Attribute                        | Grover's Algorithm                 |
|----------------------------------|------------------------------------|
| Complexity Class                 | $O(\sqrt{N})$                  |
| Classical Equivalent              | Brute-force search: $O(N)$     |
| Key Space                        | AES-256: $2^{256}$ bits        |
| Operations Needed                 | $2^{128}$                      |
| Quantum Operations/Second (Optimistic) | $10^{31}$                   |
| Required Qubits (Minimum)         | 256                                |
Limitations of Grover's Algorithm for AES-256 Attack:

1. Theoretical speedup is substantial but still requires an unfeasible number of operations $\left(2^{128}\right)$ for AES-256.

2. Quantum operations per second required $\left(10^{31}\right)$ far exceed current and near-future capabilities.

3. Quantum computer with hundreds of stable, error-corrected qubits is necessary, posing a significant technological challenge.


### Table 2: Complexity and Key Space Analysis for Shor's Algorithm

| Attribute                        | Shor's Algorithm                   |
|----------------------------------|------------------------------------|
| Complexity Class                 | $O((\log N)^2 (\log N) (\log N))$ |
| Classical Equivalent              | General Number Field Sieve: $O(\exp((\log N)^{1/3} (\log N)^{2/3}))$ |
| Key Space                        | RSA-2048: $2^{2048}$ bits      |
| Operations Needed                 | $2^{1024}$                     |
| Quantum Operations/Second (Theoretical) | N/A                              |
| Required Qubits (Minimum)         | N/A                                |


Limitations of Shor's Algorithm for AES-256 Attack:

1. Requires a quantum computer with advanced error correction and high qubit quality, currently beyond technological capabilities.
2. Theoretical speedup is significant but still requires an unfeasible number of operations $\left(2^{1024}\right)$ for RSA-2048.
3. Practical deployment of such quantum capabilities for factorizing large numbers is not expected soon, maintaining RSA's security in the short to medium term.

---

I will be adding additional details on cryptography as time permits. 



### Eni6ma.org - Copyright 2024 All Rights Reserved
#### By Dylan Rosario (Inventor of Rosario Cypher)
