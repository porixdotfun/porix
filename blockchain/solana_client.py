import base64
import json
import time
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.transaction import Transaction, AccountMeta
from solana.system_program import SYS_PROGRAM_ID, transfer, TransferParams
import solana.spl.token.instructions as spl_token
from solana.spl.token.constants import TOKEN_PROGRAM_ID

from config import CONFIG
from utils.logging import get_logger

logger = get_logger(__name__)


class SolanaClient:
    """Client to interact with the Solana blockchain"""

    def __init__(self, rpc_url=None, keypair=None):
        # Set RPC URL
        self.rpc_url = rpc_url or CONFIG.SOLANA_RPC_URL

        # Create Solana client
        self.client = Client(self.rpc_url)

        # Set keypair
        if keypair:
            if isinstance(keypair, Keypair):
                self.keypair = keypair
            else:
                # Assume keypair is a byte array or list
                self.keypair = Keypair.from_secret_key(bytes(keypair))
        else:
            # Generate a new keypair
            self.keypair = Keypair()

        logger.info(f"Initialized Solana client with RPC URL: {self.rpc_url}")
        logger.info(f"Using public key: {self.keypair.public_key}")

    def get_balance(self, public_key=None):
        """
        Get SOL balance for a public key

        Args:
            public_key: Public key to check balance for (default: this client's public key)

        Returns:
            float: Balance in SOL
        """
        if not public_key:
            public_key = self.keypair.public_key

        if isinstance(public_key, str):
            public_key = PublicKey(public_key)

        try:
            balance = self.client.get_balance(public_key)
            if "result" in balance and "value" in balance["result"]:
                # Convert lamports to SOL
                return balance["result"]["value"] / 10 ** 9
            else:
                logger.error(f"Unexpected balance response: {balance}")
                return 0
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            return 0

    def get_token_balance(self, token_account):
        """
        Get token balance for a token account

        Args:
            token_account: Token account address

        Returns:
            int: Token balance
        """
        if isinstance(token_account, str):
            token_account = PublicKey(token_account)

        try:
            info = self.client.get_token_account_balance(token_account)
            if "result" in info and "value" in info["result"]:
                return int(info["result"]["value"]["amount"])
            else:
                logger.error(f"Unexpected token balance response: {info}")
                return 0
        except Exception as e:
            logger.error(f"Error getting token balance: {str(e)}")
            return 0

    def get_token_accounts(self, owner, token_mint=None):
        """
        Get all token accounts for an owner

        Args:
            owner: Owner's public key
            token_mint: Optional token mint to filter by

        Returns:
            list: List of token accounts
        """
        if isinstance(owner, str):
            owner = PublicKey(owner)

        token_mint_filter = None
        if token_mint:
            if isinstance(token_mint, str):
                token_mint = PublicKey(token_mint)
            token_mint_filter = {"mint": str(token_mint)}

        try:
            response = self.client.get_token_accounts_by_owner(
                owner, token_mint_filter or {"programId": str(TOKEN_PROGRAM_ID)}
            )

            if "result" in response and "value" in response["result"]:
                return response["result"]["value"]
            else:
                logger.error(f"Unexpected token accounts response: {response}")
                return []
        except Exception as e:
            logger.error(f"Error getting token accounts: {str(e)}")
            return []

    def transfer_sol(self, to_pubkey, amount, priority="medium"):
        """
        Transfer SOL to another account

        Args:
            to_pubkey: Recipient's public key
            amount: Amount to transfer in SOL
            priority: Transaction priority ("low", "medium", "high")

        Returns:
            str: Transaction signature
        """
        if isinstance(to_pubkey, str):
            to_pubkey = PublicKey(to_pubkey)

        # Convert SOL to lamports
        lamports = int(amount * 10 ** 9)

        try:
            # Get recent blockhash
            blockhash_resp = self.client.get_recent_blockhash()
            blockhash = blockhash_resp["result"]["value"]["blockhash"]

            # Create transaction
            transaction = Transaction()
            transaction.recent_blockhash = blockhash

            # Add transfer instruction
            transfer_instruction = transfer(
                TransferParams(
                    from_pubkey=self.keypair.public_key,
                    to_pubkey=to_pubkey,
                    lamports=lamports
                )
            )
            transaction.add(transfer_instruction)

            # Sign and send transaction
            transaction.sign(self.keypair)

            # Set priority fee based on priority
            priority_fees = {
                "low": 5000,
                "medium": 10000,
                "high": 100000
            }
            priority_fee = priority_fees.get(priority, 10000)

            opts = TxOpts(skip_preflight=False, skip_confirmation=False, preflight_commitment="confirmed")
            resp = self.client.send_transaction(transaction, self.keypair, opts=opts)

            if "result" in resp:
                signature = resp["result"]
                logger.info(f"SOL transfer successful: {signature}")
                return signature
            else:
                logger.error(f"Failed to send SOL: {resp}")
                return None

        except Exception as e:
            logger.error(f"Error transferring SOL: {str(e)}")
            return None

    def transfer_token(self, token_mint, to_pubkey, amount, source_account=None, create_associated=True):
        """
        Transfer tokens to another account

        Args:
            token_mint: Token mint address
            to_pubkey: Recipient's public key
            amount: Amount to transfer
            source_account: Source token account (default: find owned account)
            create_associated: Whether to create associated token account if needed

        Returns:
            str: Transaction signature
        """
        if isinstance(token_mint, str):
            token_mint = PublicKey(token_mint)

        if isinstance(to_pubkey, str):
            to_pubkey = PublicKey(to_pubkey)

        try:
            # Get recent blockhash
            blockhash_resp = self.client.get_recent_blockhash()
            blockhash = blockhash_resp["result"]["value"]["blockhash"]

            # Create transaction
            transaction = Transaction()
            transaction.recent_blockhash = blockhash

            # Find source token account if not provided
            if not source_account:
                token_accounts = self.get_token_accounts(self.keypair.public_key, token_mint)
                if not token_accounts:
                    logger.error(f"No token account found for mint {token_mint}")
                    return None
                source_account = PublicKey(token_accounts[0]["pubkey"])
            elif isinstance(source_account, str):
                source_account = PublicKey(source_account)

            # Find or create destination token account
            destination_token_account = self._find_or_create_token_account(
                token_mint, to_pubkey, create_associated, transaction
            )

            if not destination_token_account:
                logger.error("Failed to find or create destination token account")
                return None

            # Add token transfer instruction
            transaction.add(
                spl_token.transfer(
                    source_account,
                    destination_token_account,
                    self.keypair.public_key,
                    amount
                )
            )

            # Sign and send transaction
            transaction.sign(self.keypair)

            opts = TxOpts(skip_preflight=False, skip_confirmation=False, preflight_commitment="confirmed")
            resp = self.client.send_transaction(transaction, self.keypair, opts=opts)

            if "result" in resp:
                signature = resp["result"]
                logger.info(f"Token transfer successful: {signature}")
                return signature
            else:
                logger.error(f"Failed to send tokens: {resp}")
                return None

        except Exception as e:
            logger.error(f"Error transferring tokens: {str(e)}")
            return None

    def _find_or_create_token_account(self, token_mint, owner, create_associated, transaction):
        """Find or create a token account for the owner"""
        try:
            # Try to find existing token account
            owner_token_accounts = self.get_token_accounts(owner, token_mint)

            if owner_token_accounts:
                # Use existing account
                return PublicKey(owner_token_accounts[0]["pubkey"])
            elif create_associated:
                # Create associated token account
                associated_token_address = self._get_associated_token_address(token_mint, owner)

                # Check if account exists
                account_info = self.client.get_account_info(associated_token_address)
                if "result" in account_info and account_info["result"]["value"] is None:
                    # Account doesn't exist, create it
                    transaction.add(
                        spl_token.create_associated_token_account(
                            self.keypair.public_key,
                            owner,
                            token_mint
                        )
                    )

                return associated_token_address
            else:
                logger.error(f"No token account found for owner {owner} and mint {token_mint}")
                return None
        except Exception as e:
            logger.error(f"Error finding or creating token account: {str(e)}")
            return None

    def _get_associated_token_address(self, token_mint, owner):
        """Get the associated token account address"""
        # This is a simplified version
        # In a real implementation, you'd use the proper SPL function
        # return spl_token.get_associated_token_address(owner, token_mint)

        # For demonstration purposes, returning a dummy address
        return PublicKey("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

    def get_transaction(self, signature):
        """
        Get transaction details

        Args:
            signature: Transaction signature

        Returns:
            dict: Transaction details
        """
        try:
            response = self.client.get_transaction(signature)
            if "result" in response:
                return response["result"]
            else:
                logger.error(f"Unexpected transaction response: {response}")
                return None
        except Exception as e:
            logger.error(f"Error getting transaction: {str(e)}")
            return None

    def get_account_info(self, public_key):
        """
        Get account information

        Args:
            public_key: Account public key

        Returns:
            dict: Account information
        """
        if isinstance(public_key, str):
            public_key = PublicKey(public_key)

        try:
            response = self.client.get_account_info(public_key)
            if "result" in response and "value" in response["result"]:
                return response["result"]["value"]
            else:
                logger.error(f"Unexpected account info response: {response}")
                return None
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None