from solana.publickey import PublicKey
from solana.keypair import Keypair
from solana.transaction import Transaction
from solana.system_program import SYS_PROGRAM_ID, create_account
import solana.spl.token.instructions as spl_token
from solana.spl.token.constants import TOKEN_PROGRAM_ID
import json
import base58
import base64

from blockchain.solana_client import SolanaClient
from utils.logging import get_logger
from config import CONFIG

logger = get_logger(__name__)


class NFTManager:
    """Manager for NFT operations"""

    def __init__(self, solana_client=None):
        self.solana_client = solana_client or SolanaClient()
        self.collection_mint = CONFIG.REPUTATION_NFT_COLLECTION

        # Initialize collection if needed
        if not self.collection_mint or self.collection_mint == "":
            logger.info("No NFT collection mint configured, collection will be created on first mint")

    def create_nft(self, recipient, metadata):
        """
        Create a new NFT for a user

        Args:
            recipient: Recipient's public key
            metadata: NFT metadata

        Returns:
            tuple: (mint_address, transaction_signature)
        """
        if isinstance(recipient, str):
            recipient = PublicKey(recipient)

        try:
            # Generate keypair for NFT mint
            nft_keypair = Keypair()

            # Get recent blockhash
            blockhash_resp = self.solana_client.client.get_recent_blockhash()
            blockhash = blockhash_resp["result"]["value"]["blockhash"]

            # Create transaction
            transaction = Transaction()
            transaction.recent_blockhash = blockhash

            # Calculate rent-exempt minimum balance
            rent_resp = self.solana_client.client.get_minimum_balance_for_rent_exemption(
                spl_token.MINT_LEN
            )
            mint_rent = rent_resp["result"]

            # Add instruction to create account for NFT mint
            transaction.add(
                create_account(
                    {
                        "from_pubkey": self.solana_client.keypair.public_key,
                        "to_pubkey": nft_keypair.public_key,
                        "lamports": mint_rent,
                        "space": spl_token.MINT_LEN,
                        "program_id": TOKEN_PROGRAM_ID
                    }
                )
            )

            # Add instruction to initialize mint (NFTs have 0 decimals)
            transaction.add(
                spl_token.initialize_mint(
                    nft_keypair.public_key,
                    0,  # 0 decimals for NFTs
                    self.solana_client.keypair.public_key,
                    None  # Freeze authority
                )
            )

            # Find or create destination token account
            token_account = self.solana_client._find_or_create_token_account(
                nft_keypair.public_key, recipient, True, transaction
            )

            if not token_account:
                logger.error("Failed to find or create token account")
                return None, None

            # Add mint instruction (amount 1 for NFT)
            transaction.add(
                spl_token.mint_to(
                    nft_keypair.public_key,
                    token_account,
                    self.solana_client.keypair.public_key,
                    1  # Mint exactly 1 token for NFT
                )
            )

            # Revoke mint authority to make the NFT non-mintable
            transaction.add(
                spl_token.set_authority(
                    nft_keypair.public_key,
                    self.solana_client.keypair.public_key,
                    None,  # New authority (None = revoke)
                    spl_token.AuthorityType.MINT_TOKENS
                )
            )

            # Set collection if available
            if self.collection_mint and self.collection_mint != "":
                # In a real implementation, this would add instructions
                # to set the collection using Metaplex standards
                pass

            # Store metadata (in a real implementation, this would use
            # Metaplex metadata standard)
            self._store_metadata(nft_keypair.public_key, metadata)

            # Sign and send transaction
            transaction.sign(self.solana_client.keypair, nft_keypair)

            resp = self.solana_client.client.send_transaction(
                transaction, self.solana_client.keypair, nft_keypair
            )

            if "result" in resp:
                signature = resp["result"]
                logger.info(f"NFT created successfully: {nft_keypair.public_key}")
                return str(nft_keypair.public_key), signature
            else:
                logger.error(f"Failed to create NFT: {resp}")
                return None, None

        except Exception as e:
            logger.error(f"Error creating NFT: {str(e)}")
            return None, None

    def create_prediction_nft(self, user_pubkey, prediction_results):
        """
        Create an NFT representing a user's prediction history

        Args:
            user_pubkey: User's public key
            prediction_results: List of prediction results

        Returns:
            tuple: (mint_address, transaction_signature)
        """
        if isinstance(user_pubkey, str):
            user_pubkey = PublicKey(user_pubkey)

        # Calculate prediction stats
        total_predictions = len(prediction_results)
        correct_predictions = sum(1 for p in prediction_results if p.get("is_correct", False))
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        # Format metadata
        metadata = {
            "name": f"Porix Predictor: {str(user_pubkey)[:6]}...{str(user_pubkey)[-4:]}",
            "description": "This NFT represents a user's prediction history on the Porix platform",
            "image": f"https://porix.io/predictor/{str(user_pubkey)}.png",  # Would be generated dynamically
            "attributes": [
                {
                    "trait_type": "Total Predictions",
                    "value": total_predictions
                },
                {
                    "trait_type": "Correct Predictions",
                    "value": correct_predictions
                },
                {
                    "trait_type": "Accuracy",
                    "value": f"{accuracy:.2%}"
                },
                {
                    "trait_type": "Rank",
                    "value": self._calculate_rank(accuracy, total_predictions)
                }
            ],
            "prediction_history": [
                {
                    "id": p.get("prediction_id"),
                    "asset": p.get("asset"),
                    "type": p.get("prediction_type"),
                    "predicted": p.get("predicted_value"),
                    "actual": p.get("actual_value"),
                    "timestamp": p.get("timestamp"),
                    "is_correct": p.get("is_correct", False)
                }
                for p in prediction_results[:10]  # Include last 10 predictions
            ]
        }

        return self.create_nft(user_pubkey, metadata)

    def _calculate_rank(self, accuracy, total_predictions):
        """Calculate user rank based on performance"""
        if total_predictions < 5:
            return "Novice"
        elif accuracy < 0.5:
            return "Apprentice"
        elif accuracy < 0.6:
            return "Adept"
        elif accuracy < 0.7:
            return "Expert"
        elif accuracy < 0.8:
            return "Master"
        else:
            return "Oracle"

    def _store_metadata(self, mint_pubkey, metadata):
        """
        Store metadata for an NFT

        In a real implementation, this would use Metaplex metadata program
        Here we're simulating storage for demonstration purposes
        """
        metadata_json = json.dumps(metadata)
        logger.info(f"Stored metadata for NFT {mint_pubkey}: {metadata_json[:100]}...")

        # This would actually call Metaplex functions to store on-chain
        # or upload to Arweave for permanent storage

    def get_nft_metadata(self, mint_address):
        """
        Get metadata for an NFT

        Args:
            mint_address: NFT mint address

        Returns:
            dict: NFT metadata
        """
        if isinstance(mint_address, str):
            mint_address = PublicKey(mint_address)

        # In a real implementation, this would fetch from Metaplex or Arweave
        # Here we'll return a placeholder

        return {
            "name": f"Porix NFT: {str(mint_address)[:6]}",
            "description": "This is a Porix platform NFT",
            "image": f"https://porix.io/nft/{str(mint_address)}.png"
        }

    def get_user_nfts(self, user_pubkey):
        """
        Get all NFTs owned by a user

        Args:
            user_pubkey: User's public key

        Returns:
            list: List of NFTs
        """
        if isinstance(user_pubkey, str):
            user_pubkey = PublicKey(user_pubkey)

        try:
            # Get all token accounts for the user
            token_accounts = self.solana_client.get_token_accounts(user_pubkey)

            nfts = []
            for account in token_accounts:
                token_account = account["pubkey"]
                account_data = account["account"]["data"]

                # Parse account data
                # In a real implementation, you'd properly decode the binary data
                # Here we're using a simplified approach

                # Get mint address from account data
                # This is a placeholder - actual implementation would decode properly
                mint_address = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

                # Get token balance
                balance = self.solana_client.get_token_balance(token_account)

                # If balance is 1, it might be an NFT
                if balance == 1:
                    # Get metadata
                    metadata = self.get_nft_metadata(mint_address)

                    nfts.append({
                        "mint": mint_address,
                        "token_account": token_account,
                        "metadata": metadata
                    })

            return nfts

        except Exception as e:
            logger.error(f"Error getting user NFTs: {str(e)}")
            return []